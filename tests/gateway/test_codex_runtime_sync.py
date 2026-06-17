import asyncio
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock


def _write_codex_config(codex_home: Path, *, model="gpt-5.5", base_url="https://api.toskaxy.xyz/v1"):
    codex_home.mkdir(parents=True, exist_ok=True)
    (codex_home / "config.toml").write_text(
        "\n".join(
            [
                'model_provider = "OpenAI"',
                f'model = "{model}"',
                'model_context_window = 1000000',
                "",
                "[model_providers.OpenAI]",
                'name = "OpenAI"',
                f'base_url = "{base_url}"',
                'wire_api = "responses"',
                "requires_openai_auth = true",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_sync_hermes_config_from_codex_updates_primary_model(tmp_path, monkeypatch):
    gateway_run = importlib.import_module("gateway.run")
    hermes_home = tmp_path / ".hermes"
    codex_home = tmp_path / ".codex"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "model:\n"
        "  provider: OpenAI\n"
        "  default: gpt-5.5\n"
        "  base_url: http://old.example/v1\n"
        "  api_mode: codex_responses\n",
        encoding="utf-8",
    )
    _write_codex_config(codex_home)
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    changed = gateway_run._sync_hermes_config_from_codex(config_home=hermes_home)

    assert changed is True
    import yaml

    cfg = yaml.safe_load((hermes_home / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["model"]["provider"] == "OpenAI"
    assert cfg["model"]["default"] == "gpt-5.5"
    assert cfg["model"]["base_url"] == "https://api.toskaxy.xyz/v1"
    assert cfg["model"]["api_mode"] == "codex_responses"
    assert cfg["model"]["context_length"] == 1000000
    assert cfg["providers"]["OpenAI"]["base_url"] == "https://api.toskaxy.xyz/v1"
    assert cfg["providers"]["OpenAI"]["transport"] == "codex_responses"
    assert cfg["providers"]["OpenAI"]["key_env"] == "OPENAI_API_KEY"


def test_sync_hermes_config_from_codex_noop_when_already_aligned(tmp_path, monkeypatch):
    gateway_run = importlib.import_module("gateway.run")
    hermes_home = tmp_path / ".hermes"
    codex_home = tmp_path / ".codex"
    hermes_home.mkdir()
    _write_codex_config(codex_home)
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    first = gateway_run._sync_hermes_config_from_codex(config_home=hermes_home)
    second = gateway_run._sync_hermes_config_from_codex(config_home=hermes_home)

    assert first is True
    assert second is False


async def _run_watcher_once(runner):
    task = asyncio.create_task(runner._codex_runtime_sync_watcher(interval=0.01))
    try:
        await asyncio.sleep(0.05)
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


def _make_runner(gateway_run):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._running = True
    runner._shutdown_event = asyncio.Event()
    runner._restart_task_started = False
    runner._restart_requested = False
    runner._background_tasks = set()
    runner.request_restart = MagicMock(return_value=True)
    return runner


def test_codex_runtime_sync_watcher_ignores_config_change(tmp_path, monkeypatch):
    gateway_run = importlib.import_module("gateway.run")
    hermes_home = tmp_path / ".hermes"
    codex_home = tmp_path / ".codex"
    hermes_home.mkdir()
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    _write_codex_config(codex_home, base_url="https://api.old.example/v1")
    gateway_run._sync_hermes_config_from_codex(config_home=hermes_home)
    runner = _make_runner(gateway_run)

    async def _exercise():
        watcher = asyncio.create_task(runner._codex_runtime_sync_watcher(interval=0.01))
        await asyncio.sleep(0.03)
        _write_codex_config(codex_home, base_url="https://api.new.example/v1")
        await asyncio.sleep(0.08)
        assert not runner.request_restart.called
        runner._shutdown_event.set()
        if not watcher.done():
            watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass

    asyncio.run(_exercise())


def test_codex_runtime_sync_watcher_restarts_on_auth_change(tmp_path, monkeypatch):
    gateway_run = importlib.import_module("gateway.run")
    hermes_home = tmp_path / ".hermes"
    codex_home = tmp_path / ".codex"
    hermes_home.mkdir()
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    _write_codex_config(codex_home)
    (codex_home / "auth.json").write_text('{"tokens":{"access_token":"a","refresh_token":"b"}}', encoding="utf-8")
    gateway_run._sync_hermes_config_from_codex(config_home=hermes_home)
    runner = _make_runner(gateway_run)

    async def _exercise():
        watcher = asyncio.create_task(runner._codex_runtime_sync_watcher(interval=0.01))
        await asyncio.sleep(0.03)
        (codex_home / "auth.json").write_text('{"tokens":{"access_token":"c","refresh_token":"d"}}', encoding="utf-8")
        await asyncio.sleep(0.08)
        assert runner.request_restart.called
        runner._shutdown_event.set()
        if not watcher.done():
            watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass

    asyncio.run(_exercise())
