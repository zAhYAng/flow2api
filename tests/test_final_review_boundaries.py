"""Integration regressions for the final model-contract/async-video review."""

import asyncio
import inspect
import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, PropertyMock, patch

from fastapi import Request

from src.api import routes
from src.core.config import Config
from src.core.database import Database
from src.core.models import GeminiGenerateContentRequest, Project, Token
from src.services.concurrency_manager import ConcurrencyManager
from src.services.flow_client import FlowClient
from src.services.generation_handler import GenerationHandler
from src.services.load_balancer import LoadBalancer
from src.services.token_manager import TokenManager


VIDEO_MODEL = "veo_3_1_t2v_fast_landscape"
JPEG_BYTES = b"\xff\xd8\xff" + b"0" * 16


class GenerationBoundaryFixture(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        for name, value in (
            ("captcha_method", "yescaptcha"), ("call_logic_mode", "polling"),
            ("extension_account_sync_enabled", False), ("cache_enabled", False), ("poll_interval", 0),
        ):
            setting = patch.object(Config, name, new_callable=PropertyMock, return_value=value)
            setting.start()
            self.addCleanup(setting.stop)
        self.db = Database(str(Path(self.temp.name) / "boundary.db"))
        await self.db.init_db()
        await self.db.init_config_from_toml({})
        self.token = Token(
            st="test-session", at="test-access", email="boundary@example.test",
            at_expires=datetime.now(timezone.utc) + timedelta(days=1),
            user_paygate_tier="PAYGATE_TIER_ONE", image_concurrency=1, video_concurrency=1,
        )
        self.token.id = await self.db.add_token(self.token)
        await self.db.add_project(Project(
            project_id="project-boundary", project_name="Boundary", token_id=self.token.id,
        ))
        self.slots = ConcurrencyManager()
        await self.slots.initialize([self.token])
        self.flow = FlowClient(proxy_manager=None, db=self.db)
        # Only external calls are substituted; account selection, ownership and persistence are real.
        self.flow.get_credits = AsyncMock(return_value={
            "credits": 100, "userPaygateTier": "PAYGATE_TIER_ONE",
        })
        self.flow.prefill_remote_browser_pool = AsyncMock()
        self.flow.generate_video_text = AsyncMock(return_value={"operations": [{
            "operation": {"name": "upstream-boundary"}, "mediaName": "media-boundary",
            "projectId": "project-boundary", "sceneId": "scene-boundary",
        }]})
        self.manager = TokenManager(self.db, self.flow)
        self.balancer = LoadBalancer(self.manager, self.slots)
        self.balancer.release_pending = AsyncMock(wraps=self.balancer.release_pending)
        self.handler = GenerationHandler(
            self.flow, self.manager, self.balancer, self.db, self.slots, None,
        )

    async def asyncTearDown(self):
        workers = list(self.handler._background_submissions.values())
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    async def pending(self):
        return await self.balancer._get_pending_count(self.token.id, False, True)


class EnqueueTokenBoundaryTests(GenerationBoundaryFixture):
    async def test_enqueue_returns_while_real_token_validation_is_blocked(self):
        entered = asyncio.Event()
        release = asyncio.Event()

        async def blocked_credits(at):
            entered.set()
            await release.wait()
            return {"credits": 73, "userPaygateTier": "PAYGATE_TIER_ONE"}

        self.flow.get_credits.side_effect = blocked_credits
        enqueue = asyncio.create_task(self.handler.enqueue_gemini_video(VIDEO_MODEL, "cat", []))
        try:
            try:
                name = await asyncio.wait_for(asyncio.shield(enqueue), timeout=0.5)
            except asyncio.TimeoutError:
                self.fail("enqueue waited for upstream TokenManager validation instead of returning a local operation")
            await asyncio.wait_for(entered.wait(), timeout=2)
            local_id = name.removeprefix("operations/")
            worker = self.handler._background_submissions[local_id]
            self.assertNotEqual(name, "operations/upstream-boundary")
            task = await self.db.get_task(local_id)
            self.assertEqual((task.status, task.upstream_operation_id), ("submitting", None))
            self.assertEqual(await self.pending(), 1)
            self.assertEqual(await self.handler.get_gemini_video_operation(name), {"name": name, "done": False})
            self.assertEqual(await self.slots.get_video_inflight(self.token.id), 0)
            release.set()
            await asyncio.wait_for(worker, timeout=2)
            task = await self.db.get_task(local_id)
            self.assertEqual((task.status, task.upstream_operation_id), ("processing", "upstream-boundary"))
            self.assertEqual((await self.db.get_token(self.token.id)).credits, 73)
            self.assertEqual(await self.pending(), 0)
            self.balancer.release_pending.assert_awaited_once_with(self.token.id, for_video_generation=True)
        finally:
            release.set()
            await asyncio.wait_for(enqueue, timeout=2)
            await asyncio.gather(*list(self.handler._background_submissions.values()), return_exceptions=True)

    async def test_default_selection_still_validates_before_owning_pending(self):
        entered = asyncio.Event()
        release = asyncio.Event()

        async def blocked_credits(at):
            entered.set()
            await release.wait()
            return {"credits": 61, "userPaygateTier": "PAYGATE_TIER_ONE"}

        self.flow.get_credits.side_effect = blocked_credits
        selecting = asyncio.create_task(self.balancer.select_token(
            for_video_generation=True, model=VIDEO_MODEL, track_pending=True,
        ))
        try:
            await asyncio.wait_for(entered.wait(), timeout=2)
            self.assertFalse(selecting.done())
            self.assertEqual(await self.pending(), 0)
        finally:
            release.set()
            selected = await asyncio.wait_for(selecting, timeout=2)
        self.assertEqual(selected.credits, 61)
        self.assertEqual(await self.pending(), 1)
        await self.balancer.release_pending(selected.id, for_video_generation=True)

    async def test_real_database_insert_failure_rolls_back_pending(self):
        # A duplicate primary key exercises SQLite's real write failure, not a mocked DB result.
        name = await self.handler.enqueue_gemini_video(VIDEO_MODEL, "first", [])
        await self.handler.wait_for_gemini_video_submission(name)
        self.balancer.release_pending.reset_mock()
        with patch("uuid.uuid4", return_value=name.removeprefix("operations/")):
            with self.assertRaisesRegex(sqlite3.IntegrityError, "UNIQUE constraint failed"):
                await self.handler.enqueue_gemini_video(VIDEO_MODEL, "duplicate", [])
        self.assertEqual(await self.pending(), 0)
        self.assertFalse(self.handler._background_submissions)
        self.balancer.release_pending.assert_awaited_once_with(self.token.id, for_video_generation=True)

    async def test_task_scheduling_failure_closes_coroutine_and_rolls_back_pending(self):
        submissions = []

        def fail_scheduling(coro):
            submissions.append(coro)
            raise RuntimeError("scheduler unavailable")

        with patch("src.services.generation_handler.asyncio.create_task", side_effect=fail_scheduling):
            with self.assertRaisesRegex(RuntimeError, "scheduler unavailable"):
                await self.handler.enqueue_gemini_video(VIDEO_MODEL, "cat", [])
        self.assertEqual(await self.pending(), 0)
        self.assertFalse(self.handler._background_submissions)
        self.assertEqual(len(submissions), 1)
        self.assertEqual(inspect.getcoroutinestate(submissions[0]), inspect.CORO_CLOSED)
        self.balancer.release_pending.assert_awaited_once_with(self.token.id, for_video_generation=True)


class UploadClassificationTests(GenerationBoundaryFixture):
    async def _assert_failed_operation(self, name, code):
        task = await self.db.get_task(name.removeprefix("operations/"))
        self.assertEqual((task.status, task.error_code), ("failed", code))
        self.assertIsNotNone(task.completed_at)
        log = await self.db.get_log_detail(task.request_log_id)
        self.assertEqual((log["status_code"], log["status_text"]), (code, "video_failed"))
        self.assertEqual(json.loads(log["response_body"])["error"]["code"], code)
        scope = {"type": "http", "scheme": "http", "headers": [(b"host", b"testserver")]}
        with patch.object(routes, "generation_handler", self.handler):
            polled = await routes.poll_video_operation(
                name.removeprefix("operations/"), raw_request=Request(scope), api_key="test",
            )
        self.assertEqual((polled["done"], polled["error"]["code"], polled["error"]["status"]), (
            True, code, "UNAVAILABLE" if code == 502 else "INTERNAL",
        ))
        self.assertEqual(await self.pending(), 0)
        self.balancer.release_pending.assert_awaited_once_with(self.token.id, for_video_generation=True)

    async def test_real_project_upload_wrapper_maps_to_502(self):
        self.flow._make_request = AsyncMock(side_effect=RuntimeError("HTTP Error 502: upstream failed"))
        self.flow.generate_video_start_image = AsyncMock()
        with patch.object(Config, "flow_max_retries", new_callable=PropertyMock, return_value=1):
            name = await self.handler.enqueue_gemini_video("omni_8s", "cat", [JPEG_BYTES])
            await self.handler.wait_for_gemini_video_submission(name)

        await self._assert_failed_operation(name, 502)
        task = await self.db.get_task(name.removeprefix("operations/"))
        self.assertIn("Project-scoped image upload failed", task.error_message)
        self.assertFalse(self.flow.generate_video_start_image.await_args_list)

    async def test_real_upload_missing_media_protocol_failure_maps_to_502(self):
        self.flow._make_request = AsyncMock(return_value={"media": {}})
        with patch.object(Config, "flow_max_retries", new_callable=PropertyMock, return_value=1):
            name = await self.handler.enqueue_gemini_video("omni_8s", "cat", [JPEG_BYTES])
            await self.handler.wait_for_gemini_video_submission(name)
        await self._assert_failed_operation(name, 502)

    async def test_upload_failure_remains_502_through_opaque_cause_wrapper(self):
        real_upload = self.flow.upload_image
        self.flow._make_request = AsyncMock(side_effect=RuntimeError("HTTP Error 502: upstream failed"))

        async def wrapped_upload(*args, **kwargs):
            try:
                return await real_upload(*args, **kwargs)
            except RuntimeError as exc:
                raise RuntimeError("Video input preparation failed") from exc

        with patch.object(Config, "flow_max_retries", new_callable=PropertyMock, return_value=1), \
                patch.object(self.flow, "upload_image", new=wrapped_upload):
            name = await self.handler.enqueue_gemini_video("omni_8s", "cat", [JPEG_BYTES])
            await self.handler.wait_for_gemini_video_submission(name)
        await self._assert_failed_operation(name, 502)

    async def test_internal_cause_chain_remains_500(self):
        async def fail_internal(**kwargs):
            try:
                raise TimeoutError("Local state lock timed out")
            except TimeoutError as exc:
                raise RuntimeError("Video input preparation failed") from exc

        self.flow.prefill_remote_browser_pool.side_effect = fail_internal
        name = await self.handler.enqueue_gemini_video(VIDEO_MODEL, "cat", [])
        await self.handler.wait_for_gemini_video_submission(name)
        await self._assert_failed_operation(name, 500)

    async def test_unclassified_internal_submission_error_remains_500(self):
        self.flow.generate_video_text = AsyncMock(side_effect=RuntimeError("local invariant violated"))
        name = await self.handler.enqueue_gemini_video(VIDEO_MODEL, "cat", [])
        await self.handler.wait_for_gemini_video_submission(name)

        await self._assert_failed_operation(name, 500)


class SynchronousPendingBoundaryTests(GenerationBoundaryFixture):
    async def _add_fallback_token(self):
        fallback = self.token.model_copy(update={
            "id": None, "st": "fallback-session", "at": "fallback-access",
            "email": "fallback@example.test",
        })
        fallback.id = await self.db.add_token(fallback)
        await self.db.add_project(Project(
            project_id="project-fallback", project_name="Fallback", token_id=fallback.id,
        ))
        await self.slots.initialize([self.token, fallback])
        return fallback

    async def test_asgi_disconnect_during_selected_token_log_cleans_pending_and_log(self):
        # An unrelated inflight slot must survive cancellation of this queued request.
        await self.slots.acquire_video(self.token.id)
        logging_started = asyncio.Event()
        real_update = self.db.update_request_log
        pending_at_disconnect = []
        messages = []

        async def paused_log_update(log_id, **kwargs):
            await real_update(log_id, **kwargs)
            if kwargs.get("status_text") == "token_selected":
                logging_started.set()
                await asyncio.Future()

        async def receive():
            await logging_started.wait()
            pending_at_disconnect.append(await self.pending())
            return {"type": "http.disconnect"}

        async def send(message):
            messages.append(message)

        scope = {
            "type": "http", "asgi": {"spec_version": "2.3"}, "method": "POST", "scheme": "http",
            "path": "/models/Veo 3.1 - Fast:generateContent", "query_string": b"",
            "headers": [(b"host", b"testserver")],
        }
        previous_tasks = asyncio.all_tasks()
        with patch.object(self.db, "update_request_log", side_effect=paused_log_update), \
                patch.object(routes, "generation_handler", self.handler):
            response = await routes.generate_content(
                model="Veo 3.1 - Fast",
                request=GeminiGenerateContentRequest(contents=[{"role": "user", "parts": [{"text": "cat"}]}]),
                raw_request=Request(scope), api_key="test",
            )
            await asyncio.wait_for(response(scope, receive, send), timeout=2)

        logs = await self.db.get_logs(include_payload=True)
        self.assertEqual(pending_at_disconnect, [1])
        self.assertEqual({
            "pending": await self.pending(),
            "releases": self.balancer.release_pending.await_count,
            "log_states": [(log["status_code"], log["status_text"]) for log in logs],
        }, {"pending": 0, "releases": 1, "log_states": [(499, "failed")]})
        self.assertIn("客户端连接已断开", json.loads(logs[0]["response_body"])["error"])
        self.assertEqual(await self.slots.get_video_inflight(self.token.id), 1)
        await self.slots.release_video(self.token.id)
        self.assertIsNone(response.body_iterator.ag_frame)
        self.assertEqual(asyncio.all_tasks() - previous_tasks, set())
        self.assertEqual(messages[0]["status"], 200)

    async def test_cancel_during_initial_log_insert_terminates_the_persisted_log(self):
        logging_started = asyncio.Event()
        finish_insert = asyncio.Event()
        real_add = self.db.add_request_log

        async def paused_insert(log):
            log_id = await real_add(log)
            if log.status_code == 102:
                logging_started.set()
                await finish_insert.wait()
            return log_id

        async def collect():
            return [chunk async for chunk in self.handler.handle_generation(VIDEO_MODEL, "cat")]

        with patch.object(self.db, "add_request_log", side_effect=paused_insert):
            generating = asyncio.create_task(collect())
            try:
                await asyncio.wait_for(logging_started.wait(), timeout=2)
                generating.cancel()
                finish_insert.set()
                with self.assertRaises(asyncio.CancelledError):
                    await generating
            finally:
                finish_insert.set()
                generating.cancel()
                await asyncio.gather(generating, return_exceptions=True)
        logs = await self.db.get_logs()
        self.assertEqual([(log["status_code"], log["status_text"]) for log in logs], [(499, "failed")])
        self.assertEqual(await self.pending(), 0)
        self.assertEqual(self.balancer.release_pending.await_count, 0)

    async def test_invalid_token_after_selection_still_releases_original_pending(self):
        real_validate = self.manager.ensure_valid_token
        validations = 0

        async def invalidate_after_selection(token):
            nonlocal validations
            validations += 1
            if validations == 1:
                return await real_validate(token)
            return None

        with patch.object(self.manager, "ensure_valid_token", side_effect=invalidate_after_selection):
            result = [chunk async for chunk in self.handler.handle_generation(VIDEO_MODEL, "cat")]
        self.assertEqual(json.loads(result[-1])["error"]["status_code"], 503)
        self.assertEqual(await self.pending(), 0)
        self.balancer.release_pending.assert_awaited_once_with(
            self.token.id, for_image_generation=False, for_video_generation=True,
        )
        self.assertEqual([(log["status_code"], log["status_text"]) for log in await self.db.get_logs()], [
            (503, "failed"),
        ])

    async def test_normal_video_fallback_releases_each_owned_pending_slot(self):
        fallback = await self._add_fallback_token()
        attempt_tokens = []
        success = self.flow.generate_video_text.return_value

        async def fail_then_succeed(**kwargs):
            attempt_tokens.append(kwargs["token_id"])
            if kwargs["token_id"] == self.token.id:
                raise RuntimeError("Flow API request failed: injected")
            return success

        self.flow.generate_video_text.side_effect = fail_then_succeed
        self.flow.check_video_status = AsyncMock(return_value={"operations": [{
            "operation": {"name": "upstream-boundary"}, "mediaName": "media-boundary",
            "status": "MEDIA_GENERATION_STATUS_SUCCESSFUL",
        }]})
        self.flow.get_media_url_redirect = AsyncMock(return_value="https://example.test/fallback.mp4")
        result = [chunk async for chunk in self.handler.handle_generation(VIDEO_MODEL, "cat")]

        self.assertEqual(attempt_tokens, [self.token.id, fallback.id])
        self.assertIn("https://example.test/fallback.mp4", json.loads(result[-1])["choices"][0]["message"]["content"])
        self.assertEqual((await self.db.get_task("upstream-boundary")).status, "completed")
        self.assertEqual((await self.db.get_token(fallback.id)).use_count, 1)
        self.assertEqual((await self.db.get_token(self.token.id)).use_count, 0)
        self.assertEqual([(log["status_code"], log["status_text"]) for log in await self.db.get_logs()], [
            (200, "completed"), (200, "completed"),
        ])
        self.assertEqual(await self.balancer._get_pending_count(fallback.id, False, True), 0)
        self.assertEqual(await self.pending(), 0)
        self.assertEqual(self.balancer.release_pending.await_count, 2)
        self.assertEqual(await self.slots.get_video_inflight(self.token.id), 0)
        self.assertEqual(await self.slots.get_video_inflight(fallback.id), 0)

    async def _assert_fallback_validation_cancel(self, is_image):
        fallback = await self._add_fallback_token()
        validating = asyncio.Event()
        real_validate = self.manager.ensure_valid_token
        fallback_validations = 0

        async def blocked_fallback_validation(token):
            nonlocal fallback_validations
            if token.id == fallback.id:
                fallback_validations += 1
                if fallback_validations == 2:
                    validating.set()
                    await asyncio.Future()
            return await real_validate(token)

        self.flow.generate_video_text.side_effect = RuntimeError("Flow API request failed: injected")
        self.flow.generate_image = AsyncMock(side_effect=RuntimeError("Flow API request failed: injected"))
        model = "gemini-3.0-pro-image-landscape" if is_image else VIDEO_MODEL

        async def collect():
            return [chunk async for chunk in self.handler.handle_generation(model, "cat")]

        with patch.object(self.manager, "ensure_valid_token", side_effect=blocked_fallback_validation):
            generating = asyncio.create_task(collect())
            try:
                await asyncio.wait_for(validating.wait(), timeout=2)
                self.assertEqual(await self.balancer._get_pending_count(fallback.id, is_image, not is_image), 1)
            finally:
                generating.cancel()
                result = await asyncio.gather(generating, return_exceptions=True)
        self.assertIsInstance(result[0], asyncio.CancelledError)
        self.assertEqual(await self.balancer._get_pending_count(fallback.id, is_image, not is_image), 0)
        self.assertEqual(await self.balancer._get_pending_count(self.token.id, is_image, not is_image), 0)
        released_ids = [call.args[0] for call in self.balancer.release_pending.await_args_list]
        self.assertCountEqual(released_ids, [self.token.id, fallback.id])
        self.assertTrue(all(log["status_code"] != 102 for log in await self.db.get_logs()))

    async def test_cancel_during_image_fallback_validation_releases_new_owner(self):
        await self._assert_fallback_validation_cancel(is_image=True)

    async def test_cancel_during_video_fallback_validation_releases_new_owner(self):
        await self._assert_fallback_validation_cancel(is_image=False)

    async def test_cancel_during_fallback_generation_terminates_both_logs(self):
        fallback = await self._add_fallback_token()
        generating_fallback = asyncio.Event()

        async def fail_then_block(**kwargs):
            if kwargs["token_id"] == self.token.id:
                raise RuntimeError("Flow API request failed: injected")
            generating_fallback.set()
            await asyncio.Future()

        self.flow.generate_video_text.side_effect = fail_then_block

        async def collect():
            return [chunk async for chunk in self.handler.handle_generation(VIDEO_MODEL, "cat")]

        generating = asyncio.create_task(collect())
        try:
            await asyncio.wait_for(generating_fallback.wait(), timeout=2)
        finally:
            generating.cancel()
            result = await asyncio.gather(generating, return_exceptions=True)
        self.assertIsInstance(result[0], asyncio.CancelledError)
        logs = await self.db.get_logs()
        self.assertCountEqual([(log["status_code"], log["status_text"]) for log in logs], [
            (499, "failed"), (499, "failed"),
        ])
        self.assertEqual(await self.balancer._get_pending_count(fallback.id, False, True), 0)
        self.assertEqual(await self.pending(), 0)
        self.assertEqual(self.balancer.release_pending.await_count, 2)


class ModelPersistenceBoundaryTests(GenerationBoundaryFixture):
    async def _assert_actual_model_persists(self, model, images, expected_model, method_name):
        self.flow._make_request = AsyncMock(return_value={"media": {"name": "input-media"}})
        submitted_models = []

        async def submit(**kwargs):
            submitted_models.append(kwargs["model_key"])
            return {"operations": [{
                "operation": {"name": "upstream-boundary"}, "mediaName": "media-boundary",
                "projectId": "project-boundary", "sceneId": "scene-boundary",
            }]}

        setattr(self.flow, method_name, submit)
        name = await self.handler.enqueue_gemini_video(model, "cat", images)
        await self.handler.wait_for_gemini_video_submission(name)
        task = await self.db.get_task(name.removeprefix("operations/"))
        self.assertEqual(submitted_models, [expected_model])
        self.assertEqual((task.status, task.model), ("processing", expected_model))
        self.flow.check_video_status = AsyncMock(return_value={"operations": [{
            "operation": {"name": "upstream-boundary"}, "mediaName": "media-boundary",
            "status": "MEDIA_GENERATION_STATUS_SUCCESSFUL",
        }]})
        self.flow.get_media_url_redirect = AsyncMock(return_value="https://example.test/video.mp4")
        completed = await self.handler.get_gemini_video_operation(name)
        self.assertTrue(completed["done"])
        task = await self.db.get_task(name.removeprefix("operations/"))
        self.assertEqual((task.status, task.model), ("completed", expected_model))
        log = await self.db.get_log_detail(task.request_log_id)
        self.assertEqual((log["status_code"], log["status_text"]), (200, "completed"))
        self.assertEqual(json.loads(log["request_body"])["model"], expected_model)
        response = json.loads(log["response_body"])
        self.assertEqual(response["model"], expected_model)
        self.assertEqual(response["generated_assets"]["model"], expected_model)
        self.assertEqual(await self.pending(), 0)

    async def test_omni_first_frame_persists_i2v_model_in_task_and_completion_log(self):
        await self._assert_actual_model_persists(
            "omni_8s", [JPEG_BYTES], "abra_i2v_8s", "generate_video_start_image",
        )

    async def test_omni_start_end_persists_i2v_model_in_task_and_completion_log(self):
        await self._assert_actual_model_persists(
            "omni_4s", [JPEG_BYTES, JPEG_BYTES], "abra_i2v_4s", "generate_video_start_end",
        )


if __name__ == "__main__":
    unittest.main()
