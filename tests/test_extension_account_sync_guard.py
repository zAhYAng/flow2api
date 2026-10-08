import asyncio
import unittest
from unittest.mock import AsyncMock, PropertyMock, patch

from src.core.config import Config
from src.services.token_manager import TokenManager


class ExtensionAccountSyncGuardTests(unittest.TestCase):
    def test_disabled_extension_account_sync_sends_no_browser_message(self):
        manager = TokenManager(object(), object())
        service = AsyncMock()
        mock_get_instance = AsyncMock(return_value=service)

        with (
            patch.object(
                Config,
                "extension_account_sync_enabled",
                new_callable=PropertyMock,
                return_value=False,
            ),
            patch(
                "src.services.browser_captcha_extension.ExtensionCaptchaService.get_instance",
                new=mock_get_instance,
            ),
        ):
            asyncio.run(manager._request_extension_account_sync(1, "at_refresh_failed"))

        mock_get_instance.assert_not_awaited()
        service.request_account_sync.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
