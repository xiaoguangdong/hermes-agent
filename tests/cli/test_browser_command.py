from unittest.mock import MagicMock, patch

from cli import HermesCLI


class TestBrowserCommand:
    def _make_cli(self):
        cli_obj = HermesCLI.__new__(HermesCLI)
        cli_obj.config = {}
        cli_obj.console = MagicMock()
        cli_obj.agent = None
        cli_obj.conversation_history = []
        cli_obj.session_id = None
        cli_obj._pending_input = MagicMock()
        cli_obj._app = None
        cli_obj._last_invalidate = 0.0
        cli_obj._command_running = False
        cli_obj._command_status = ""
        return cli_obj

    def test_browser_status_uses_probe_output_for_unreachable_target(self, capsys):
        cli_obj = self._make_cli()

        with patch.dict("os.environ", {"BROWSER_CDP_URL": "http://127.0.0.1:9222"}, clear=False), \
             patch("tools.browser_tool.probe_cdp_endpoint", return_value={
                 "ok": False,
                 "status": "unreachable",
                 "version_url": "http://127.0.0.1:9222/json/version",
                 "resolved_url": "",
                 "message": "Cannot reach CDP endpoint: refused",
             }):
            cli_obj.process_command("/browser status")

        output = capsys.readouterr().out
        assert "Browser: connected to live Chrome via CDP" in output
        assert "Status: ⚠ not reachable" in output
        assert "Cannot reach CDP endpoint: refused" in output
        assert "Discovery URL: http://127.0.0.1:9222/json/version" in output

    def test_browser_connect_says_target_saved_when_probe_not_ready(self, capsys):
        cli_obj = self._make_cli()

        with patch.dict("os.environ", {}, clear=False), \
             patch.object(cli_obj, "_try_launch_chrome_debug", return_value=False), \
             patch("tools.browser_tool.cleanup_all_browsers"), \
             patch("tools.browser_tool.probe_cdp_endpoint", return_value={
                 "ok": False,
                 "status": "unreachable",
                 "version_url": "http://127.0.0.1:9222/json/version",
                 "resolved_url": "",
                 "message": "Cannot reach CDP endpoint: refused",
             }):
            cli_obj.process_command("/browser connect")

        output = capsys.readouterr().out
        assert "Browser target saved for live Chrome via CDP" in output
        assert "Note: target was saved, but live CDP connectivity is not ready yet" in output

    def test_browser_connect_reports_reachable_and_resolved_websocket(self, capsys):
        cli_obj = self._make_cli()
        ws_url = "ws://127.0.0.1:9222/devtools/browser/abc"

        with patch.dict("os.environ", {}, clear=False), \
             patch("tools.browser_tool.cleanup_all_browsers"), \
             patch("tools.browser_tool.probe_cdp_endpoint", return_value={
                 "ok": True,
                 "status": "ready",
                 "version_url": "http://127.0.0.1:9222/json/version",
                 "resolved_url": ws_url,
                 "message": "CDP discovery endpoint is reachable",
             }):
            cli_obj.process_command("/browser connect")

        output = capsys.readouterr().out
        assert "Chrome is already reachable" in output
        assert "Resolved WebSocket: ws://127.0.0.1:9222/devtools/browser/abc" in output
        assert "Browser connected to live Chrome via CDP" in output
