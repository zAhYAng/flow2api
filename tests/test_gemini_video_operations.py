import asyncio
import inspect
import json
import tempfile
import sqlite3
import base64
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import routes
from src.core.auth import verify_api_key_flexible
from src.core.database import Database
from src.core.models import Task, Token
from src.services.generation_handler import GenerationHandler
from src.services.load_balancer import LoadBalancer


class GeminiVideoApiTests(unittest.TestCase):
    def setUp(self):
        self.previous = routes.generation_handler
        app = FastAPI()
        app.include_router(routes.router)
        app.dependency_overrides[verify_api_key_flexible] = lambda: "test"
        self.client = TestClient(app)

    def tearDown(self):
        routes.generation_handler = self.previous
        self.client.close()

    def test_submit_returns_pollable_operation_without_waiting(self):
        routes.generation_handler = SimpleNamespace(enqueue_gemini_video=AsyncMock(return_value="operations/local-123"))
        response = self.client.post("/v1beta/models/Veo%203.1%20-%20Fast:predictLongRunning", json={
            "instances": [{"prompt": "a cat runs"}],
            "parameters": {"aspectRatio": "16:9"},
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"name": "operations/local-123", "done": False})
        self.assertEqual(response.content, b'{"name":"operations/local-123","done":false}')
        self.assertEqual(response.headers["content-type"], "application/json")
        call = routes.generation_handler.enqueue_gemini_video.await_args.kwargs
        self.assertEqual(call["model"], "veo_3_1_t2v_fast_landscape")
        self.assertEqual(call["prompt"], "a cat runs")

    def test_public_veo_duration_maps_to_matching_real_model(self):
        expected = {
            ("Veo 3.1 - Quality", 4): "veo_3_1_t2v_4s",
            ("Veo 3.1 - Quality", 6): "veo_3_1_t2v_6s",
            ("Veo 3.1 - Quality", 8): "veo_3_1_t2v_8s",
            ("Veo 3.1 - Fast", 4): "veo_3_1_t2v_fast_4s",
            ("Veo 3.1 - Fast", 6): "veo_3_1_t2v_fast_6s",
            ("Veo 3.1 - Fast", 8): "veo_3_1_t2v_fast_8s",
            ("Veo 3.1 - Lite", 4): "veo_3_1_t2v_lite_4s_landscape",
            ("Veo 3.1 - Lite", 6): "veo_3_1_t2v_lite_6s_landscape",
            ("Veo 3.1 - Lite", 8): "veo_3_1_t2v_lite_8s_landscape",
        }
        for (model, seconds), internal_model in expected.items():
            with self.subTest(model=model, seconds=seconds):
                self.handler = SimpleNamespace(enqueue_gemini_video=AsyncMock(return_value="operations/op-123"))
                routes.generation_handler = self.handler
                response = self.client.post(f"/v1beta/models/{model}:predictLongRunning", json={
                    "instances": [{"prompt": "a cat runs"}],
                    "parameters": {"aspectRatio": "16:9", "durationSeconds": seconds},
                })
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json(), {"name": "operations/op-123", "done": False})
                self.assertEqual(self.handler.enqueue_gemini_video.await_args.kwargs["model"], internal_model)

    def test_poll_returns_gemini_video_response(self):
        routes.generation_handler = SimpleNamespace(get_gemini_video_operation=AsyncMock(return_value={
            "name": "operations/op-123", "done": True,
            "response": {"generateVideoResponse": {"generatedSamples": [{
                "video": {"uri": "https://example.test/video.mp4"},
            }]}},
        }))
        response = self.client.get("/v1beta/operations/op-123")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["response"]["generateVideoResponse"]["generatedSamples"][0]["video"]["uri"],
                         "https://example.test/video.mp4")

    def test_poll_completion_exposes_direct_uri_for_declarative_agent(self):
        routes.generation_handler = SimpleNamespace(get_gemini_video_operation=AsyncMock(return_value={
            "name": "operations/op-123", "done": True,
            "response": {"generateVideoResponse": {"generatedSamples": [{
                "video": {"uri": "https://example.test/video.mp4"},
            }]}},
        }))
        response = self.client.get("/v1beta/operations/op-123")
        sample = response.json()["response"]["generateVideoResponse"]["generatedSamples"][0]
        self.assertEqual(sample["uri"], "https://example.test/video.mp4")
        self.assertEqual(sample["video"]["uri"], "https://example.test/video.mp4")

    def test_model_scoped_operation_can_be_polled(self):
        routes.generation_handler = SimpleNamespace(get_gemini_video_operation=AsyncMock(return_value={
            "name": "operations/op-123", "done": False,
        }))
        response = self.client.get("/v1beta/models/Veo%203.1%20-%20Fast/operations/op-123")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"name": "operations/op-123", "done": False})

    def test_poll_returns_absolute_uri_for_cached_video(self):
        routes.generation_handler = SimpleNamespace(get_gemini_video_operation=AsyncMock(return_value={
            "name": "operations/op-123", "done": True,
            "response": {"generateVideoResponse": {"generatedSamples": [{"video": {"uri": "/tmp/video.mp4"}}]}},
        }))
        response = self.client.get("/v1beta/operations/op-123")
        self.assertEqual(response.json()["response"]["generateVideoResponse"]["generatedSamples"][0]["video"]["uri"],
                         "http://testserver/tmp/video.mp4")

    def test_image_submission_routes_to_first_frame_generation(self):
        routes.generation_handler = SimpleNamespace(enqueue_gemini_video=AsyncMock(return_value="operations/op-123"))
        response = self.client.post("/v1beta/models/Veo%203.1%20-%20Fast:predictLongRunning", json={
            "instances": [{"prompt": "animate", "image": {
                "bytesBase64Encoded": base64.b64encode(b"image").decode(), "mimeType": "image/png",
            }}],
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"name": "operations/op-123", "done": False})
        kwargs = routes.generation_handler.enqueue_gemini_video.await_args.kwargs
        self.assertEqual(kwargs["images"], [b"image"])
        self.assertEqual(kwargs["model"], "veo_3_1_i2v_s_fast_fl")

    def test_agent_nested_inline_data_image_is_accepted(self):
        routes.generation_handler = SimpleNamespace(enqueue_gemini_video=AsyncMock(return_value="operations/op-123"))
        image = base64.b64encode(b"agent-image").decode()
        response = self.client.post("/v1beta/models/Veo%203.1%20-%20Quality:predictLongRunning", json={
            "instances": [{"prompt": "animate", "image": {
                "inlineData": {"mimeType": "image/png", "data": image},
            }}],
            "parameters": {"durationSeconds": 4, "aspectRatio": "16:9"},
        })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"name": "operations/op-123", "done": False})
        kwargs = routes.generation_handler.enqueue_gemini_video.await_args.kwargs
        self.assertEqual(kwargs["images"], [b"agent-image"])
        self.assertEqual(kwargs["model"], "veo_3_1_i2v_s_4s")

    def test_agent_first_and_last_frames_are_both_submitted(self):
        routes.generation_handler = SimpleNamespace(enqueue_gemini_video=AsyncMock(return_value="operations/op-123"))
        response = self.client.post("/v1beta/models/Veo%203.1%20-%20Quality:predictLongRunning", json={
            "instances": [{
                "prompt": "transition between frames",
                "image": {"inlineData": {
                    "mimeType": "image/png", "data": base64.b64encode(b"first-frame").decode(),
                }},
                "lastFrame": {"inlineData": {
                    "mimeType": "image/png", "data": base64.b64encode(b"last-frame").decode(),
                }},
            }],
            "parameters": {"durationSeconds": 4, "aspectRatio": "16:9"},
        })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"name": "operations/op-123", "done": False})
        kwargs = routes.generation_handler.enqueue_gemini_video.await_args.kwargs
        self.assertEqual(kwargs["images"], [b"first-frame", b"last-frame"])
        self.assertEqual(kwargs["model"], "veo_3_1_i2v_s_4s")

    def test_agent_file_uri_image_is_downloaded_and_accepted(self):
        routes.generation_handler = SimpleNamespace(enqueue_gemini_video=AsyncMock(return_value="operations/op-123"))
        with patch("src.api.routes._load_image_bytes_from_uri", new=AsyncMock(return_value=b"remote-image")) as loader:
            response = self.client.post("/v1beta/models/Veo%203.1%20-%20Fast:predictLongRunning", json={
                "instances": [{"prompt": "animate", "image": {
                    "fileUri": "https://example.test/frame.png", "mimeType": "image/png",
                }}],
                "parameters": {"durationSeconds": 6},
            })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"name": "operations/op-123", "done": False})
        loader.assert_awaited_once_with("https://example.test/frame.png")
        self.assertEqual(routes.generation_handler.enqueue_gemini_video.await_args.kwargs["images"], [b"remote-image"])
        self.assertEqual(routes.generation_handler.enqueue_gemini_video.await_args.kwargs["model"],
                         "veo_3_1_i2v_s_fast_6s_fl")

    def test_unknown_operation_returns_gemini_not_found(self):
        routes.generation_handler = SimpleNamespace(get_gemini_video_operation=AsyncMock(return_value=None))
        response = self.client.get("/v1beta/operations/unknown")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["status"], "NOT_FOUND")

    def test_public_video_models_advertise_long_running_method(self):
        response = self.client.get("/v1beta/models")
        veo = next(m for m in response.json()["models"] if m["name"] == "models/Veo 3.1 - Fast")
        self.assertIn("predictLongRunning", veo["supportedGenerationMethods"])

    def test_unsupported_internal_modes_are_not_advertised(self):
        response = self.client.get("/models/internal")
        model = next(m for m in response.json()["models"] if m["name"] == "models/veo_3_1_extend")
        self.assertNotIn("predictLongRunning", model["supportedGenerationMethods"])

    def test_standard_google_veo_name_submits(self):
        routes.generation_handler = SimpleNamespace(enqueue_gemini_video=AsyncMock(return_value="operations/op-123"))
        response = self.client.post("/v1beta/models/veo-3.1-fast-generate-preview:predictLongRunning", json={
            "instances": [{"prompt": "a cat runs"}],
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"name": "operations/op-123", "done": False})
        self.assertEqual(routes.generation_handler.enqueue_gemini_video.await_args.kwargs["model"],
                         "veo_3_1_t2v_fast_landscape")

    def test_background_submission_failure_is_reported_by_polling(self):
        failed_operation = {
            "name": "operations/local-123", "done": True,
            "error": {"code": 502, "message": "upstream model rejected image"},
        }
        routes.generation_handler = SimpleNamespace(
            enqueue_gemini_video=AsyncMock(return_value="operations/local-123"),
            get_gemini_video_operation=AsyncMock(return_value=failed_operation),
        )
        response = self.client.post("/v1beta/models/Veo%203.1%20-%20Quality:predictLongRunning", json={
            "instances": [{"prompt": "animate"}],
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"name": "operations/local-123", "done": False})

        polled = self.client.get(f"/v1beta/{response.json()['name']}")
        self.assertEqual(polled.status_code, 200)
        self.assertEqual(polled.json(), failed_operation)
        routes.generation_handler.get_gemini_video_operation.assert_awaited_once_with("operations/local-123")

    def test_model_access_denied_is_reported_by_polling(self):
        failed_operation = {
            "name": "operations/local-123", "done": True,
            "error": {"code": 403, "message": "Selected Flow account cannot access this video model"},
        }
        routes.generation_handler = SimpleNamespace(
            enqueue_gemini_video=AsyncMock(return_value="operations/local-123"),
            get_gemini_video_operation=AsyncMock(return_value=failed_operation),
        )
        response = self.client.post("/v1beta/models/Veo%203.1%20-%20Quality:predictLongRunning", json={
            "instances": [{"prompt": "animate"}],
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"name": "operations/local-123", "done": False})

        polled = self.client.get(f"/v1beta/{response.json()['name']}")
        self.assertEqual(polled.status_code, 200)
        self.assertEqual(polled.json(), failed_operation)
        routes.generation_handler.get_gemini_video_operation.assert_awaited_once_with("operations/local-123")

    def test_enqueue_without_available_account_returns_service_unavailable(self):
        routes.generation_handler = SimpleNamespace(
            enqueue_gemini_video=AsyncMock(side_effect=ValueError("No available video token")),
        )
        response = self.client.post("/v1beta/models/Veo%203.1%20-%20Fast:predictLongRunning", json={
            "instances": [{"prompt": "animate"}],
        })
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"error": {
            "code": 503, "message": "No available video token", "status": "UNAVAILABLE",
        }})

    def test_enqueue_failure_returns_internal_error_and_logs_exception(self):
        routes.generation_handler = SimpleNamespace(
            enqueue_gemini_video=AsyncMock(side_effect=RuntimeError("local operation persistence failed")),
        )
        with self.assertLogs("src.api.routes", level="ERROR") as captured:
            response = self.client.post("/v1beta/models/Veo%203.1%20-%20Fast:predictLongRunning", json={
                "instances": [{"prompt": "animate"}],
            })
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"error": {
            "code": 500, "message": "Video enqueue failed: local operation persistence failed", "status": "INTERNAL",
        }})
        self.assertTrue(any("RuntimeError: local operation persistence failed" in line for line in captured.output))

    def test_explicit_model_rejects_conflicting_aspect_ratio(self):
        routes.generation_handler = SimpleNamespace(enqueue_gemini_video=AsyncMock())
        response = self.client.post("/v1beta/models/veo_3_1_t2v_fast_landscape:predictLongRunning", json={
            "instances": [{"prompt": "a cat runs"}], "parameters": {"aspectRatio": "9:16"},
        })
        self.assertEqual(response.status_code, 400)
        routes.generation_handler.enqueue_gemini_video.assert_not_awaited()

    def test_invalid_models_or_inputs_do_not_submit(self):
        routes.generation_handler = SimpleNamespace(enqueue_gemini_video=AsyncMock())
        for model, body in (
            ("Nano Banana 2", {"instances": [{"prompt": "cat"}]}),
            ("Veo 3.1 - Fast", {"instances": [{"prompt": " "}]}),
            ("Veo 3.1 - Fast", {"instances": [{"prompt": "cat"}], "parameters": {"durationSeconds": 10}}),
            ("Veo 3.1 - Fast", {"instances": [{"prompt": "cat"}], "parameters": {"aspectRatio": "1:1"}}),
            ("Veo 3.1 - Fast", {"instances": [{"prompt": "cat"}], "parameters": {"sampleCount": 2}}),
        ):
            with self.subTest(model=model, body=body):
                response = self.client.post(f"/v1beta/models/{model}:predictLongRunning", json=body)
                self.assertEqual(response.status_code, 400)
        routes.generation_handler.enqueue_gemini_video.assert_not_awaited()

    def test_polling_requires_api_key(self):
        self.client.app.dependency_overrides.clear()
        self.assertIn(self.client.get("/v1beta/operations/op-123").status_code, (401, 403))
        self.assertIn(self.client.post("/v1beta/models/Veo%203.1%20-%20Fast:predictLongRunning", json={
            "instances": [{"prompt": "cat"}],
        }).status_code, (401, 403))


class GeminiJsonKeepAliveTests(unittest.IsolatedAsyncioTestCase):
    async def test_closing_stream_cancels_and_awaits_pending_generation(self):
        started = asyncio.Event()
        cleaned_up = asyncio.Event()
        generation_task = None

        async def generate():
            nonlocal generation_task
            generation_task = asyncio.current_task()
            started.set()
            try:
                await asyncio.Future()
            finally:
                await asyncio.sleep(0)
                cleaned_up.set()

        stream = routes._stream_json_with_keep_alive(generate())
        try:
            self.assertEqual(await anext(stream), b" ")
            await asyncio.wait_for(started.wait(), timeout=1.0)
            await stream.aclose()
            self.assertTrue(cleaned_up.is_set(), "Stream close must await generation cleanup")
            self.assertTrue(generation_task.cancelled())
        finally:
            if generation_task is not None:
                generation_task.cancel()
                await asyncio.gather(generation_task, return_exceptions=True)
            await stream.aclose()

    async def test_cancelling_stream_cancels_and_awaits_pending_generation(self):
        started = asyncio.Event()
        cleaned_up = asyncio.Event()
        generation_task = None
        next_chunk = None

        async def generate():
            nonlocal generation_task
            generation_task = asyncio.current_task()
            started.set()
            try:
                await asyncio.Future()
            finally:
                await asyncio.sleep(0)
                cleaned_up.set()

        stream = routes._stream_json_with_keep_alive(generate())
        try:
            self.assertEqual(await anext(stream), b" ")
            await asyncio.wait_for(started.wait(), timeout=1.0)
            next_chunk = asyncio.create_task(anext(stream))
            await asyncio.sleep(0)
            next_chunk.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await next_chunk
            self.assertTrue(cleaned_up.is_set(), "Stream cancellation must await generation cleanup")
            self.assertTrue(generation_task.cancelled())
        finally:
            if next_chunk is not None:
                next_chunk.cancel()
                await asyncio.gather(next_chunk, return_exceptions=True)
            if generation_task is not None:
                generation_task.cancel()
                await asyncio.gather(generation_task, return_exceptions=True)
            await stream.aclose()

    async def test_completed_generation_preserves_response_serialization(self):
        for result, expected in (
            (b'{"done":true}', b'{"done":true}'),
            ('{"text":"猫"}', '{"text":"猫"}'.encode("utf-8")),
            ({"text": "猫"}, '{"text": "猫"}'.encode("utf-8")),
        ):
            with self.subTest(result=result):
                async def generate():
                    return result

                chunks = [chunk async for chunk in routes._stream_json_with_keep_alive(generate())]
                self.assertEqual(chunks, [b" ", expected])


class GeminiVideoPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Database(str(Path(self.temp.name) / "flow.db"))
        await self.db.init_db()
        token = Token(st="st", at="at", email="test@example.com", user_paygate_tier="PAYGATE_TIER_ONE")
        token.id = await self.db.add_token(token)
        self.token = token
        self.flow = SimpleNamespace(
            prefill_remote_browser_pool=AsyncMock(),
            upload_image=AsyncMock(side_effect=["first-media", "last-media"]),
            generate_video_text=AsyncMock(return_value={"operations": [{
                "operation": {"name": "op-123"}, "mediaName": "media-123", "projectId": "project-1",
            }]}),
            generate_video_start_image=AsyncMock(return_value={"operations": [{
                "operation": {"name": "op-123"}, "mediaName": "media-123", "projectId": "project-1",
            }]}),
            generate_video_start_end=AsyncMock(return_value={"operations": [{
                "operation": {"name": "op-123"}, "mediaName": "media-123", "projectId": "project-1",
            }]}),
            check_video_status=AsyncMock(return_value={"operations": [{
                "operation": {"name": "op-123"}, "mediaName": "media-123", "projectId": "project-1",
                "status": "MEDIA_GENERATION_STATUS_ACTIVE",
            }]}),
            get_media_url_redirect=AsyncMock(return_value="https://example.test/video.mp4"),
        )
        self.manager = SimpleNamespace(
            get_active_tokens=AsyncMock(return_value=[token]),
            needs_at_refresh=lambda token: False,
            ensure_valid_token=AsyncMock(return_value=token),
            ensure_project_exists=AsyncMock(return_value="project-1"),
            record_usage=AsyncMock(), record_success=AsyncMock(),
        )
        balancer_config = patch("src.services.load_balancer.config", SimpleNamespace(
            captcha_method="yescaptcha", call_logic_mode="default",
        ))
        balancer_config.start()
        self.addCleanup(balancer_config.stop)
        self.balancer = LoadBalancer(self.manager)
        # 保留真实计数副作用，同时检测重复释放（真实计数会将负数截断为零）。
        self.balancer.release_pending = AsyncMock(wraps=self.balancer.release_pending)
        self.handler = GenerationHandler(self.flow, self.manager, self.balancer, self.db, None, None)

    async def asyncTearDown(self):
        self.temp.cleanup()

    async def _assert_failed_operation(self, name, error_code):
        """核对真实任务、轮询结果和持久化日志的失败语义。"""
        task = await self.db.get_task(name.removeprefix("operations/"))
        self.assertEqual(task.status, "failed")
        self.assertEqual(task.error_code, error_code)
        self.assertIsNotNone(task.completed_at)
        error = {"code": error_code, "message": task.error_message}
        self.assertEqual(await self.handler.get_gemini_video_operation(name), {
            "name": name, "done": True, "error": error,
        })
        self.assertIsNotNone(task.request_log_id)
        log = await self.db.get_log_detail(task.request_log_id)
        self.assertIsNotNone(log)
        self.assertEqual(log["operation"], "generate_video")
        self.assertEqual(log["status_code"], error_code)
        self.assertEqual(log["status_text"], "video_failed")
        self.assertEqual(log["progress"], 0)
        request = json.loads(log["request_body"])
        self.assertEqual(request["operation_name"], name)
        self.assertEqual(request.get("protocol"), "gemini_predictLongRunning")
        response = json.loads(log["response_body"])
        self.assertEqual(response.get("name"), name)
        self.assertEqual(response["status"], "failed")
        self.assertEqual(response["error"], error)
        self.assertEqual(await self.balancer._get_pending_count(self.token.id, False, True), 0)

    async def test_operation_survives_new_handler_and_polls_upstream(self):
        name = await self.handler.submit_gemini_video(model="veo_3_1_t2v_fast_landscape", prompt="a cat", images=[])
        self.assertEqual(name, "operations/op-123")
        self.assertEqual(self.flow.check_video_status.await_count, 0)
        restarted = GenerationHandler(self.flow, self.manager, self.balancer, self.db, None, None)
        pending = await restarted.get_gemini_video_operation(name)
        self.assertEqual(pending, {"name": name, "done": False})
        self.flow.check_video_status.return_value["operations"][0]["status"] = "MEDIA_GENERATION_STATUS_SUCCESSFUL"
        finished = await restarted.get_gemini_video_operation(name)
        self.assertTrue(finished["done"])
        self.assertEqual(finished["response"]["generateVideoResponse"]["generatedSamples"][0]["video"]["uri"],
                         "https://example.test/video.mp4")
        self.manager.record_usage.assert_awaited_once_with(self.token.id, is_video=True)

    async def test_two_images_submit_start_end_generation(self):
        name = await self.handler.submit_gemini_video(
            model="veo_3_1_i2v_s_4s", prompt="transition", images=[b"first", b"last"],
        )
        self.assertEqual(name, "operations/op-123")
        self.assertEqual(self.flow.upload_image.await_count, 2)
        self.flow.generate_video_start_end.assert_awaited_once()
        kwargs = self.flow.generate_video_start_end.await_args.kwargs
        self.assertEqual(kwargs["model_key"], "veo_3_1_i2v_s_quality_4s_fl")
        self.assertEqual(kwargs["start_media_id"], "first-media")
        self.assertEqual(kwargs["end_media_id"], "last-media")
        self.flow.generate_video_start_image.assert_not_awaited()

    async def test_old_database_task_schema_migrates(self):
        path = Path(self.temp.name) / "legacy.db"
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE tasks (id INTEGER PRIMARY KEY, task_id TEXT UNIQUE NOT NULL, token_id INTEGER NOT NULL, model TEXT NOT NULL, prompt TEXT NOT NULL, status TEXT NOT NULL, progress INTEGER, result_urls TEXT, error_message TEXT, scene_id TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, completed_at TIMESTAMP)")
        migrated = Database(str(path))
        await migrated.init_db()
        await migrated.check_and_migrate_db()
        with sqlite3.connect(path) as connection:
            columns = {row[1] for row in connection.execute("PRAGMA table_info(tasks)")}
        self.assertTrue({"project_id", "media_name"}.issubset(columns))
        self.assertIn("upstream_operation_id", columns)
        self.assertIn("error_code", columns)

    async def test_failed_upstream_operation_stays_failed_across_polls(self):
        name = await self.handler.submit_gemini_video(model="veo_3_1_t2v_fast_landscape", prompt="a cat", images=[])
        self.flow.check_video_status.return_value["operations"][0].update({
            "status": "MEDIA_GENERATION_STATUS_FAILED", "operation": {"name": "op-123", "error": {"message": "rejected"}},
        })
        failed = await self.handler.get_gemini_video_operation(name)
        self.assertEqual(failed["error"]["message"], "rejected")
        self.assertTrue(failed["done"])
        await self.handler.get_gemini_video_operation(name)
        self.assertEqual(self.flow.check_video_status.await_count, 1)

    async def test_finished_video_with_no_url_can_be_polled_again(self):
        name = await self.handler.submit_gemini_video(model="veo_3_1_t2v_fast_landscape", prompt="cat", images=[])
        operation = self.flow.check_video_status.return_value["operations"][0]
        operation["status"] = "MEDIA_GENERATION_STATUS_SUCCESSFUL"
        self.flow.get_media_url_redirect.return_value = None
        self.flow.get_media = AsyncMock(return_value={"video": {"encodedVideo": ""}})
        self.assertEqual(await self.handler.get_gemini_video_operation(name), {"name": name, "done": False})
        self.flow.get_media_url_redirect.return_value = "https://example.test/video.mp4"
        self.assertTrue((await self.handler.get_gemini_video_operation(name))["done"])

    async def test_completed_cached_video_can_recover_after_cache_cleanup(self):
        name = await self.handler.submit_gemini_video(model="veo_3_1_t2v_fast_landscape", prompt="cat", images=[])
        self.flow.check_video_status.return_value["operations"][0]["status"] = "MEDIA_GENERATION_STATUS_SUCCESSFUL"
        self.flow.get_media_url_redirect.return_value = None
        self.flow.get_media = AsyncMock(return_value={"video": {"encodedVideo": base64.b64encode(b"mp4").decode()}})
        self.handler.file_cache.cache_dir = Path(self.temp.name)
        finished = await self.handler.get_gemini_video_operation(name)
        self.assertTrue(finished["done"])
        cached = self.handler.file_cache.cache_dir / finished["response"]["generateVideoResponse"]["generatedSamples"][0]["video"]["uri"].split("/")[-1]
        cached.unlink()
        refreshed = await self.handler.get_gemini_video_operation(name)
        self.assertTrue(refreshed["done"])
        self.assertNotEqual(finished, refreshed)

    async def test_enqueue_returns_local_operation_and_submits_in_background(self):
        """入队立即返回本地 operation，后台异步提交到 Flow"""
        name = await self.handler.enqueue_gemini_video(model="veo_3_1_t2v_fast_landscape", prompt="cat", images=[])
        self.assertTrue(name.startswith("operations/"))
        local_id = name.replace("operations/", "")
        pending = await self.handler.get_gemini_video_operation(name)
        self.assertEqual(pending["done"], False)
        await self.handler.wait_for_gemini_video_submission(name)
        task = await self.db.get_task(local_id)
        self.assertEqual(task.upstream_operation_id, "op-123")
        self.assertEqual(task.status, "processing")

    async def test_enqueue_with_permission_denied_persists_error_code(self):
        """后台提交遇到权限拒绝时持久化 error_code"""
        self.flow.generate_video_text.side_effect = ValueError("Account tier does not support this video model")
        name = await self.handler.enqueue_gemini_video(model="veo_3_1_t2v_fast_landscape", prompt="cat", images=[])
        await self.handler.wait_for_gemini_video_submission(name)
        local_id = name.replace("operations/", "")
        task = await self.db.get_task(local_id)
        self.assertEqual(task.status, "failed")
        self.assertEqual(task.error_code, 403)
        self.assertIn("tier", task.error_message.lower())
        await self._assert_failed_operation(name, 403)

    async def test_real_load_balancer_tracks_pending_until_background_completes(self):
        release_submission = asyncio.Event()
        pending_at_persistence = []
        real_create_task = self.db.create_task

        async def create_task(task):
            pending_at_persistence.append(
                await self.balancer._get_pending_count(self.token.id, False, True)
            )
            return await real_create_task(task)

        async def hold_submission(**kwargs):
            await release_submission.wait()

        self.flow.prefill_remote_browser_pool.side_effect = hold_submission
        with patch.object(self.db, "create_task", side_effect=create_task):
            name = await self.handler.enqueue_gemini_video(
                model="veo_3_1_t2v_fast_landscape", prompt="cat", images=[],
            )
        local_id = name.removeprefix("operations/")
        background_task = self.handler._background_submissions[local_id]
        try:
            self.assertEqual(pending_at_persistence, [1])
            self.assertEqual(await self.balancer._get_pending_count(self.token.id, False, True), 1)
            self.assertEqual(await self.handler.get_gemini_video_operation(name), {"name": name, "done": False})
            self.assertEqual((await self.db.get_task(local_id)).status, "submitting")
        finally:
            release_submission.set()
            await asyncio.wait_for(background_task, timeout=2.0)

        task = await self.db.get_task(local_id)
        self.assertEqual(task.status, "processing")
        self.assertNotEqual(local_id, "op-123")
        self.assertEqual(task.upstream_operation_id, "op-123")
        self.assertEqual(await self.balancer._get_pending_count(self.token.id, False, True), 0)
        self.balancer.release_pending.assert_awaited_once_with(self.token.id, for_video_generation=True)

    async def test_enqueue_rolls_back_pending_before_task_persistence(self):
        with patch.object(self.handler, "_resolve_video_model_key_for_tier", side_effect=RuntimeError("Model resolution failed")):
            with self.assertRaisesRegex(RuntimeError, "Model resolution failed"):
                await self.handler.enqueue_gemini_video(
                    model="veo_3_1_t2v_fast_landscape", prompt="cat", images=[],
                )
        self.assertEqual(await self.balancer._get_pending_count(self.token.id, False, True), 0)
        self.balancer.release_pending.assert_awaited_once_with(self.token.id, for_video_generation=True)
        self.assertFalse(self.handler._background_submissions)

    async def test_enqueue_rolls_back_pending_if_task_persistence_fails(self):
        error = RuntimeError("Task insert failed")
        with patch.object(self.db, "create_task", side_effect=error):
            with self.assertRaises(RuntimeError) as raised:
                await self.handler.enqueue_gemini_video(
                    model="veo_3_1_t2v_fast_landscape", prompt="cat", images=[],
                )
        self.assertIs(raised.exception, error)
        self.assertEqual(await self.balancer._get_pending_count(self.token.id, False, True), 0)
        self.balancer.release_pending.assert_awaited_once_with(self.token.id, for_video_generation=True)
        self.assertFalse(self.handler._background_submissions)

    async def test_enqueue_rolls_back_pending_if_background_task_creation_fails(self):
        submissions = []

        def fail_task_creation(submission):
            submissions.append(submission)
            raise RuntimeError("Background task creation failed")

        try:
            with patch("src.services.generation_handler.asyncio.create_task", side_effect=fail_task_creation):
                with self.assertRaisesRegex(RuntimeError, "Background task creation failed"):
                    await self.handler.enqueue_gemini_video(
                        model="veo_3_1_t2v_fast_landscape", prompt="cat", images=[],
                    )
            self.assertEqual(await self.balancer._get_pending_count(self.token.id, False, True), 0)
            self.balancer.release_pending.assert_awaited_once_with(self.token.id, for_video_generation=True)
            self.assertFalse(self.handler._background_submissions)
            self.assertEqual(len(submissions), 1)
            self.assertEqual(inspect.getcoroutinestate(submissions[0]), inspect.CORO_CLOSED)
        finally:
            for submission in submissions:
                submission.close()

    async def test_enqueue_cancellation_during_persistence_releases_pending(self):
        persistence_started = asyncio.Event()
        hold_persistence = asyncio.Event()

        async def blocked_create_task(task):
            persistence_started.set()
            await hold_persistence.wait()

        with patch.object(self.db, "create_task", side_effect=blocked_create_task):
            enqueue_task = asyncio.create_task(self.handler.enqueue_gemini_video(
                model="veo_3_1_t2v_fast_landscape", prompt="cat", images=[],
            ))
            try:
                await asyncio.wait_for(persistence_started.wait(), timeout=2.0)
                pending_before_cancel = await self.balancer._get_pending_count(self.token.id, False, True)
            finally:
                enqueue_task.cancel()
                results = await asyncio.gather(enqueue_task, return_exceptions=True)
        self.assertIsInstance(results[0], asyncio.CancelledError)
        self.assertEqual(pending_before_cancel, 1)
        self.assertEqual(await self.balancer._get_pending_count(self.token.id, False, True), 0)
        self.balancer.release_pending.assert_awaited_once_with(self.token.id, for_video_generation=True)
        self.assertFalse(self.handler._background_submissions)

    async def test_enqueue_without_available_token_does_not_release_pending(self):
        self.manager.get_active_tokens.return_value = []
        with self.assertRaisesRegex(ValueError, "No available video token"):
            await self.handler.enqueue_gemini_video(
                model="veo_3_1_t2v_fast_landscape", prompt="cat", images=[],
            )
        self.assertEqual(await self.balancer._get_pending_count(self.token.id, False, True), 0)
        self.balancer.release_pending.assert_not_awaited()

    async def test_enqueue_with_upstream_failure_persists_gemini_error(self):
        """后台提交遇到普通上游失败时持久化 Gemini 错误"""
        self.flow.generate_video_text.side_effect = RuntimeError("Video submission returned no operation")
        name = await self.handler.enqueue_gemini_video(model="veo_3_1_t2v_fast_landscape", prompt="cat", images=[])
        await self.handler.wait_for_gemini_video_submission(name)
        local_id = name.replace("operations/", "")
        task = await self.db.get_task(local_id)
        self.assertEqual(task.status, "failed")
        self.assertEqual(task.error_code, 502)
        failed = await self.handler.get_gemini_video_operation(name)
        self.assertTrue(failed["done"])
        self.assertIn("error", failed)
        self.assertEqual(failed["error"]["code"], 502)
        await self._assert_failed_operation(name, 502)

    async def test_enqueue_creates_request_log_and_stores_log_id(self):
        """异步提交成功时必须创建 request log，记录本地 operation name"""
        name = await self.handler.enqueue_gemini_video(model="veo_3_1_t2v_fast_landscape", prompt="cat", images=[])
        await self.handler.wait_for_gemini_video_submission(name)
        
        local_id = name.replace("operations/", "")
        task = await self.db.get_task(local_id)
        
        # 必须有 request_log_id
        self.assertIsNotNone(task.request_log_id, "request_log_id must be set")
        
        log = await self.db.get_log_detail(task.request_log_id)
        self.assertEqual(log["status_code"], 102)
        self.assertEqual(log["status_text"], "video_submitting")
        self.assertEqual(log["progress"], 25)
        request = json.loads(log["request_body"])
        self.assertEqual(request["operation_name"], name)
        self.assertEqual(request["protocol"], "gemini_predictLongRunning")
        self.assertEqual(request["upstream_operation_id"], "op-123")

    async def test_release_pending_called_exactly_once_on_success(self):
        """成功提交路径必须恰好 release_pending 一次"""
        self.balancer.release_pending.reset_mock()
        name = await self.handler.enqueue_gemini_video(model="veo_3_1_t2v_fast_landscape", prompt="cat", images=[])
        await self.handler.wait_for_gemini_video_submission(name)
        
        # 应该被调用 1 次（后台任务中）
        # release_pending(token_id, for_video_generation=True)
        self.assertEqual(self.balancer.release_pending.await_count, 1)
        call_kwargs = self.balancer.release_pending.await_args.kwargs
        self.assertEqual(call_kwargs.get('for_video_generation'), True)
        self.assertEqual(await self.balancer._get_pending_count(self.token.id, False, True), 0)

    async def test_release_pending_called_exactly_once_on_permission_denied(self):
        """权限拒绝路径必须恰好 release_pending 一次"""
        self.balancer.release_pending.reset_mock()
        self.flow.generate_video_text.side_effect = ValueError("Account tier does not support this video model")
        name = await self.handler.enqueue_gemini_video(model="veo_3_1_t2v_fast_landscape", prompt="cat", images=[])
        await self.handler.wait_for_gemini_video_submission(name)
        
        self.assertEqual(self.balancer.release_pending.await_count, 1)

        await self._assert_failed_operation(name, 403)

    async def test_release_pending_called_exactly_once_on_upstream_failure(self):
        """上游失败路径必须恰好 release_pending 一次"""
        self.balancer.release_pending.reset_mock()
        self.flow.generate_video_text.side_effect = RuntimeError("Upstream API error")
        name = await self.handler.enqueue_gemini_video(model="veo_3_1_t2v_fast_landscape", prompt="cat", images=[])
        await self.handler.wait_for_gemini_video_submission(name)
        
        self.assertEqual(self.balancer.release_pending.await_count, 1)
        self.assertEqual(await self.balancer._get_pending_count(self.token.id, False, True), 0)

    async def test_error_code_403_for_permission_denied(self):
        """权限/层级拒绝必须使用 error_code=403"""
        for error in (
            RuntimeError("Flow API request failed: PUBLIC_ERROR_MODEL_ACCESS_DENIED"),
            RuntimeError("Flow API request failed: PERMISSION_DENIED"),
            ValueError("PERMISSION_DENIED: Selected account cannot access the model"),
        ):
            with self.subTest(error=str(error)):
                self.flow.generate_video_text.side_effect = error
                name = await self.handler.enqueue_gemini_video(
                    model="veo_3_1_t2v_fast_landscape", prompt="cat", images=[],
                )
                await self.handler.wait_for_gemini_video_submission(name)
                await self._assert_failed_operation(name, 403)

    async def test_error_code_503_for_invalid_token(self):
        """无效 Token 必须使用 error_code=503"""
        # 选择账号时有效，后台重新校验时失效。
        self.manager.ensure_valid_token.side_effect = [self.token, None]
        name = await self.handler.enqueue_gemini_video(model="veo_3_1_t2v_fast_landscape", prompt="cat", images=[])
        await self.handler.wait_for_gemini_video_submission(name)
        
        local_id = name.replace("operations/", "")
        task = await self.db.get_task(local_id)
        self.assertEqual(task.error_code, 503)
        await self._assert_failed_operation(name, 503)

    async def test_error_code_502_for_upstream_api_failure(self):
        """上游 API 失败（无 operations、无 operation ID）必须使用 error_code=502"""
        for result in (
            {"operations": []},
            {"operations": [{"operation": {}, "mediaName": "media-123", "projectId": "project-1"}]},
        ):
            with self.subTest(result=result):
                self.flow.generate_video_text.return_value = result
                name = await self.handler.enqueue_gemini_video(
                    model="veo_3_1_t2v_fast_landscape", prompt="cat", images=[],
                )
                await self.handler.wait_for_gemini_video_submission(name)
                await self._assert_failed_operation(name, 502)

    async def test_wait_exposes_background_task_exceptions(self):
        """wait_for_gemini_video_submission 必须暴露后台异常，不能只超时"""
        self.flow.generate_video_text.side_effect = RuntimeError("Flow API crashed")
        name = await self.handler.enqueue_gemini_video(model="veo_3_1_t2v_fast_landscape", prompt="cat", images=[])
        
        # 等待应该快速完成而不是超时，并且后续查询应显示失败状态
        await self.handler.wait_for_gemini_video_submission(name)
        
        local_id = name.replace("operations/", "")
        task = await self.db.get_task(local_id)
        self.assertEqual(task.status, "failed")
        self.assertIsNotNone(task.error_code)

    async def test_submitting_to_restart_interrupted_reads_fresh_state(self):
        """后台 task 清理后重读状态，防止竞态中断"""
        name = await self.handler.enqueue_gemini_video(model="veo_3_1_t2v_fast_landscape", prompt="cat", images=[])
        local_id = name.replace("operations/", "")
        
        # 等待完成提交
        await self.handler.wait_for_gemini_video_submission(name)
        
        # 此时任务应已转为 processing（成功提交） 或 failed
        task = await self.db.get_task(local_id)
        self.assertNotEqual(task.status, "submitting")
        
        # 现在模拟手动清理后台任务字典并标记为 submitting（重启场景）
        self.handler._background_submissions.pop(local_id, None)
        await self.db.update_task(local_id, status="submitting")
        
        # 轮询应该检测重启中断
        result = await self.handler.get_gemini_video_operation(name)
        self.assertTrue(result["done"])
        self.assertEqual(result["error"]["code"], 503)


    async def test_wait_exposes_uncaught_background_exception_immediately(self):
        """wait 必须快速暴露后台未捕获异常，不能只轮询 DB"""
        local_id = "uncaught-submission"
        name = f"operations/{local_id}"
        await self.db.create_task(Task(
            task_id=local_id, token_id=self.token.id,
            model="veo_3_1_t2v_fast_landscape", prompt="cat", status="submitting",
        ))

        async def failing_coroutine():
            try:
                await asyncio.sleep(0)
                raise RuntimeError("Simulated unhandled background error")
            finally:
                self.handler._background_submissions.pop(local_id, None)

        failing_task = asyncio.create_task(failing_coroutine())
        self.handler._background_submissions[local_id] = failing_task
        start = asyncio.get_running_loop().time()
        try:
            with self.assertRaisesRegex(RuntimeError, "Simulated unhandled background error"):
                await asyncio.wait_for(
                    self.handler.wait_for_gemini_video_submission(name, timeout=30.0),
                    timeout=2.0,
                )
            self.assertLess(asyncio.get_running_loop().time() - start, 2.0)
        finally:
            if not failing_task.done():
                failing_task.cancel()
            await asyncio.gather(failing_task, return_exceptions=True)
            self.handler._background_submissions.pop(local_id, None)

    async def test_wait_timeout_does_not_cancel_background_submission(self):
        """等待超时后提交仍存活，并能完成真实的数据库状态转换。"""
        local_id = "slow-submission"
        name = f"operations/{local_id}"
        await self.db.create_task(Task(
            task_id=local_id, token_id=self.token.id,
            model="veo_3_1_t2v_fast_landscape", prompt="cat", status="submitting",
        ))
        release = asyncio.Event()

        async def slow_submission():
            try:
                await release.wait()
                await self.db.update_task(local_id, status="processing", upstream_operation_id="op-123")
            finally:
                self.handler._background_submissions.pop(local_id, None)

        background_task = asyncio.create_task(slow_submission())
        self.handler._background_submissions[local_id] = background_task
        try:
            with self.assertRaises(TimeoutError):
                await self.handler.wait_for_gemini_video_submission(name, timeout=0.01)
            self.assertFalse(background_task.done())
            release.set()
            await asyncio.wait_for(background_task, timeout=2.0)
            self.assertEqual((await self.db.get_task(local_id)).status, "processing")
        finally:
            release.set()
            if not background_task.done():
                background_task.cancel()
            await asyncio.gather(background_task, return_exceptions=True)
            self.handler._background_submissions.pop(local_id, None)

    async def test_get_operation_reads_fresh_state_when_bg_task_missing(self):
        """get_operation 读取到 submitting 后，后台任务不存在时必须重新读取 DB"""
        name = await self.handler.enqueue_gemini_video(model="veo_3_1_t2v_fast_landscape", prompt="cat", images=[])
        local_id = name.replace("operations/", "")

        # 等待提交完成
        await self.handler.wait_for_gemini_video_submission(name)

        fresh_task = await self.db.get_task(local_id)
        self.assertEqual(fresh_task.status, "processing")
        self.assertNotIn(local_id, self.handler._background_submissions)
        stale_task = fresh_task.model_copy(update={
            "status": "submitting", "upstream_operation_id": None,
            "project_id": None, "media_name": None, "request_log_id": None, "progress": 0,
        })
        snapshots = iter((stale_task, fresh_task))
        real_get_task = self.db.get_task

        async def read_snapshot(task_id):
            try:
                return next(snapshots)
            except StopIteration:
                return await real_get_task(task_id)

        # 前两次读分别模拟提交前后的快照，其余操作仍使用真实 SQLite。
        with patch.object(self.db, "get_task", side_effect=read_snapshot):
            result = await self.handler.get_gemini_video_operation(name)

        self.assertEqual(result, {"name": name, "done": False})
        task_after = await self.db.get_task(local_id)
        self.assertEqual(task_after.status, "processing")
        self.assertEqual(task_after.upstream_operation_id, "op-123")
        self.assertIsNone(task_after.error_code)

    async def test_invalid_image_count_returns_400_error_code(self):
        """验证不支持图片数量在后台路径得到 error_code=400"""
        name = await self.handler.enqueue_gemini_video(
            model="veo_3_1_t2v_fast_landscape", prompt="cat", images=[b"first", b"last"],
        )
        await self.handler.wait_for_gemini_video_submission(name)

        local_id = name.replace("operations/", "")
        task = await self.db.get_task(local_id)
        self.assertEqual(task.error_code, 400, "Image count validation should return 400, not 403")
        await self._assert_failed_operation(name, 400)

    async def test_missing_input_image_returns_400_and_logs_failure(self):
        name = await self.handler.enqueue_gemini_video(model="veo_3_1_i2v_s_4s", prompt="cat", images=[])
        await self.handler.wait_for_gemini_video_submission(name)
        await self._assert_failed_operation(name, 400)

    async def test_missing_account_returns_503_and_logs_failure(self):
        # 只模拟后台读取不到账号；真实删除会级联删除任务，无法验证失败持久化。
        with patch.object(self.db, "get_token", new=AsyncMock(return_value=None)):
            name = await self.handler.enqueue_gemini_video(
                model="veo_3_1_t2v_fast_landscape", prompt="cat", images=[],
            )
            await self.handler.wait_for_gemini_video_submission(name)
        await self._assert_failed_operation(name, 503)

    async def test_account_tier_rejection_returns_403_and_logs_failure(self):
        ultra_token = self.token.model_copy(update={"user_paygate_tier": "PAYGATE_TIER_TWO"})
        self.manager.get_active_tokens.return_value = [ultra_token]
        self.manager.ensure_valid_token.side_effect = [ultra_token, self.token]
        name = await self.handler.enqueue_gemini_video(
            model="veo_3_1_t2v_fast_ultra_relaxed", prompt="cat", images=[],
        )
        await self.handler.wait_for_gemini_video_submission(name)
        await self._assert_failed_operation(name, 403)

    async def test_failed_paths_create_request_log_with_correct_codes(self):
        """失败路径（503/502）必须创建 request log"""
        # 测试 503 路径
        self.manager.ensure_valid_token.side_effect = [self.token, None]
        name1 = await self.handler.enqueue_gemini_video(model="veo_3_1_t2v_fast_landscape", prompt="cat", images=[])
        await self.handler.wait_for_gemini_video_submission(name1)

        local_id1 = name1.replace("operations/", "")
        task1 = await self.db.get_task(local_id1)
        self.assertIsNotNone(task1.request_log_id, "503 failure must have request_log_id")
        self.assertEqual(task1.error_code, 503)

        # 测试 502 路径
        self.flow.generate_video_text.side_effect = None  # 重置
        self.manager.ensure_valid_token = AsyncMock(return_value=self.token)  # 恢复
        self.flow.generate_video_text.return_value = {"operations": []}  # 无 operations

        name2 = await self.handler.enqueue_gemini_video(model="veo_3_1_t2v_fast_landscape", prompt="cat", images=[])
        await self.handler.wait_for_gemini_video_submission(name2)

        local_id2 = name2.replace("operations/", "")
        task2 = await self.db.get_task(local_id2)
        self.assertIsNotNone(task2.request_log_id, "502 failure must have request_log_id")
        self.assertEqual(task2.error_code, 502)
        await self._assert_failed_operation(name1, 503)
        await self._assert_failed_operation(name2, 502)

    async def test_submission_exceptions_persist_classified_error_and_log(self):
        cases = (
            (ValueError("Invalid video parameter"), 400),
            (RuntimeError("Flow API request failed: upstream unavailable"), 502),
            (ValueError("Flow API request failed: invalid upstream response"), 502),
            (RuntimeError("Flow API text request failed: upstream timeout"), 502),
            (RuntimeError("Video submission returned no operation ID"), 502),
            (ValueError("Token not found"), 503),
            (RuntimeError("Unexpected internal state"), 500),
            (RuntimeError("Unexpected support registry state"), 500),
        )
        for error, error_code in cases:
            with self.subTest(error=str(error), error_code=error_code):
                self.flow.generate_video_text.side_effect = error
                name = await self.handler.enqueue_gemini_video(
                    model="veo_3_1_t2v_fast_landscape", prompt="cat", images=[],
                )
                await self.handler.wait_for_gemini_video_submission(name)
                await self._assert_failed_operation(name, error_code)

    async def test_flow_video_api_timeout_returns_502_and_logs_failure(self):
        error = Exception("Flow video API request timed out after 45s")
        self.flow.generate_video_text.side_effect = error
        name = await self.handler.enqueue_gemini_video(
            model="veo_3_1_t2v_fast_landscape", prompt="cat", images=[],
        )
        await self.handler.wait_for_gemini_video_submission(name)
        await self._assert_failed_operation(name, 502)
        self.assertEqual((await self.db.get_task(name.removeprefix("operations/"))).error_message, str(error))

    async def test_local_database_timeout_returns_500_and_logs_failure(self):
        with patch.object(self.db, "get_token", side_effect=TimeoutError("Local database read timed out")):
            name = await self.handler.enqueue_gemini_video(
                model="veo_3_1_t2v_fast_landscape", prompt="cat", images=[],
            )
            await self.handler.wait_for_gemini_video_submission(name)
        await self._assert_failed_operation(name, 500)
        task = await self.db.get_task(name.removeprefix("operations/"))
        self.assertEqual(task.error_message, "Local database read timed out")



if __name__ == "__main__":
    unittest.main()
