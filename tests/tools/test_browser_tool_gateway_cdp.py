from contextvars import Context
from types import SimpleNamespace

from gateway.session_context import clear_session_vars, set_session_vars
from tools.browser_tool import _get_cdp_override


def test_get_cdp_override_prefers_gateway_session_context_over_empty_env(monkeypatch):
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)

    def _fake_get(url, timeout=10):
        assert url == "http://127.0.0.1:9222/json/version"
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/browser/test-session"
            },
        )

    monkeypatch.setattr("tools.browser_tool.requests.get", _fake_get)
    tokens = set_session_vars(browser_cdp_url="http://127.0.0.1:9222")
    try:
        assert _get_cdp_override() == "ws://127.0.0.1:9222/devtools/browser/test-session"
    finally:
        clear_session_vars(tokens)


def test_get_cdp_override_falls_back_to_process_env(monkeypatch):
    def _fake_get(url, timeout=10):
        assert url == "http://127.0.0.1:9333/json/version"
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "webSocketDebuggerUrl": "ws://127.0.0.1:9333/devtools/browser/from-env"
            },
        )

    monkeypatch.setattr("tools.browser_tool.requests.get", _fake_get)
    monkeypatch.setenv("BROWSER_CDP_URL", "http://127.0.0.1:9333")
    assert Context().run(_get_cdp_override) == "ws://127.0.0.1:9333/devtools/browser/from-env"
