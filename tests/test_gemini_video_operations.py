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
from src.core.models import Token
from src.services.generation_handler import GenerationHandler


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
        routes.generation_handler = SimpleNamespace(submit_gemini_video=AsyncMock(return_value="operations/op-123"))
        response = self.client.post("/v1beta/models/Veo%203.1%20-%20Fast:predictLongRunning", json={
            "instances": [{"prompt": "a cat runs"}],
            "parameters": {"aspectRatio": "16:9"},
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"name": "operations/op-123", "done": False})
        call = routes.generation_handler.submit_gemini_video.await_args.kwargs
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
                self.handler = SimpleNamespace(submit_gemini_video=AsyncMock(return_value="operations/op-123"))
                routes.generation_handler = self.handler
                response = self.client.post(f"/v1beta/models/{model}:predictLongRunning", json={
                    "instances": [{"prompt": "a cat runs"}],
                    "parameters": {"aspectRatio": "16:9", "durationSeconds": seconds},
                })
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(self.handler.submit_gemini_video.await_args.kwargs["model"], internal_model)

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
            "response": {"generateVideoResponse": {"generatedSamples": [{
                "video": {"uri": "/cache/video.mp4"},
            }]}},
        }))
        response = self.client.get("/v1beta/operations/op-123")
        uri = response.json()["response"]["generateVideoResponse"]["generatedSamples"][0]["video"]["uri"]
        self.assertTrue(uri.startswith("http"))


class GeminiVideoPersistenceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.NamedTemporaryFile(delete=False)
        self.temp.close()
        self.db = Database(self.temp.name)
        self.token = Token(id=1, st="st-test", email="test@example.com")
        self.flow = SimpleNamespace(
            generate_video=AsyncMock(return_value={"operationName": "op-123", "sceneId": "scene-456"}),
            generate_video_start_image=AsyncMock(return_value={"operationName": "op-123", "sceneId": "scene-456"}),
            generate_video_start_end=AsyncMock(return_value={"operationName": "op-123", "sceneId": "scene-456"}),
            upload_image=AsyncMock(side_effect=lambda image, _: f"{image.decode()}-media"),
            check_video_status=AsyncMock(return_value={"operations": [{"status": "MEDIA_GENERATION_STATUS_PROCESSING", "sceneId": "scene-456", "mediaId": "media-789"}]}),
            get_media_url_redirect=AsyncMock(return_value="https://example.test/video.mp4"),
        )
        self.manager = SimpleNamespace(
            get_available_token_for_video=AsyncMock(return_value=self.token),
            record_usage=AsyncMock(),
            record_error=AsyncMock(),
        )
        self.balancer = SimpleNamespace(flow_client=AsyncMock(return_value=self.flow))
        self.handler = GenerationHandler(self.flow, self.manager, self.balancer, self.db, None, None)

    def tearDown(self):
        Path(self.temp.name).unlink(missing_ok=True)

    async def test_submit_persists_task_with_scene_id(self):
        await self.db.init_db()
        name = await self.handler.submit_gemini_video(model="veo_3_1_t2v_fast_landscape", prompt="a cat", images=[])
        task = await self.db.get_task(name)
        self.assertEqual(task.task_id, name)
        self.assertEqual(task.scene_id, "scene-456")
        self.assertEqual(task.model, "veo_3_1_t2v_fast_landscape")
        self.assertEqual(task.prompt, "a cat")

    async def test_resumed_handler_loads_persisted_task(self):
        await self.db.init_db()
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
        legacy_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        legacy_temp.close()
        path = Path(legacy_temp.name)
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
        try:
            path.unlink()
        except PermissionError:
            pass

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


if __name__ == "__main__":
    unittest.main()
