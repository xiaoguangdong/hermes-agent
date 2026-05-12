from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key
from gateway.session_context import get_session_env


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._background_tasks = set()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    runner._browser_cdp_urls = {}

    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner.session_store._generate_session_key.return_value = build_session_key(_make_source())
    return runner


@pytest.mark.asyncio
async def test_browser_command_dispatches_to_handler():
    runner = _make_runner()
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("/browser leaked through to the agent")
    )
    runner._handle_browser_command = AsyncMock(return_value="browser ok")

    result = await runner._handle_message(_make_event("/browser status"))

    assert result == "browser ok"
    runner._handle_browser_command.assert_awaited_once()
    runner._run_agent.assert_not_called()


@pytest.mark.asyncio
async def test_browser_connect_stores_cdp_url_per_session(monkeypatch):
    runner = _make_runner()

    monkeypatch.setattr(
        "tools.browser_tool.probe_cdp_endpoint",
        lambda url: {
            "ok": True,
            "status": "ready",
            "resolved_url": "ws://127.0.0.1:9222/devtools/browser/test-session",
            "version_url": "http://127.0.0.1:9222/json/version",
            "message": "CDP discovery endpoint is reachable",
        },
    )

    result = await runner._handle_browser_command(
        _make_event("/browser connect http://127.0.0.1:9222")
    )

    session_key = build_session_key(_make_source())
    assert runner._browser_cdp_urls[session_key] == "http://127.0.0.1:9222"
    assert result is not None
    assert "9222" in result
    assert "Status: ✓ reachable" in result
    assert "Resolved WebSocket:" in result


@pytest.mark.asyncio
async def test_browser_connect_reports_saved_target_when_unreachable(monkeypatch):
    runner = _make_runner()

    monkeypatch.setattr(
        "tools.browser_tool.probe_cdp_endpoint",
        lambda url: {
            "ok": False,
            "status": "unreachable",
            "resolved_url": "",
            "version_url": "http://127.0.0.1:9222/json/version",
            "message": "Cannot reach CDP endpoint: boom",
        },
    )

    result = await runner._handle_browser_command(
        _make_event("/browser connect http://127.0.0.1:9222")
    )

    assert "target saved" in result.lower()
    assert "Status: ⚠ not reachable" in result
    assert "Note: target was saved" in result
    assert "Discovery URL:" in result


@pytest.mark.asyncio
async def test_browser_disconnect_clears_session_cdp_url():
    runner = _make_runner()
    session_key = build_session_key(_make_source())
    runner._browser_cdp_urls[session_key] = "http://127.0.0.1:9222"

    result = await runner._handle_browser_command(_make_event("/browser disconnect"))

    assert session_key not in runner._browser_cdp_urls
    assert "disconnected" in result.lower()


@pytest.mark.asyncio
async def test_browser_status_reports_discovery_failure_details(monkeypatch):
    runner = _make_runner()
    session_key = build_session_key(_make_source())
    runner._browser_cdp_urls[session_key] = "http://127.0.0.1:9222"

    monkeypatch.setattr(
        "tools.browser_tool.probe_cdp_endpoint",
        lambda url: {
            "ok": False,
            "status": "port_open_discovery_failed",
            "resolved_url": "",
            "version_url": "http://127.0.0.1:9222/json/version",
            "message": "TCP port is reachable but CDP discovery failed: boom",
        },
    )

    result = await runner._handle_browser_command(_make_event("/browser status"))

    assert "Status: ⚠ port reachable but CDP discovery failed" in result
    assert "Discovery URL:" in result


def test_set_session_env_exposes_browser_cdp_url():
    runner = _make_runner()
    source = _make_source()
    session_key = build_session_key(source)
    runner._browser_cdp_urls[session_key] = "http://127.0.0.1:9222"
    context = SimpleNamespace(source=source, session_key=session_key)

    from gateway.run import GatewayRunner

    tokens = GatewayRunner._set_session_env(runner, context)
    try:
        assert get_session_env("BROWSER_CDP_URL") == "http://127.0.0.1:9222"
    finally:
        GatewayRunner._clear_session_env(runner, tokens)
