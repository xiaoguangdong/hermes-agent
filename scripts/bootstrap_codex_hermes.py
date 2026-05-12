#!/usr/bin/env python3
"""Bootstrap a local Hermes home from the user's Codex CLI config."""

from __future__ import annotations

import argparse
import json
import os
import stat
import tomllib
from pathlib import Path

import yaml


DEFAULT_MODEL = "gpt-5.4"
DEFAULT_PROVIDER = "getrouter"
DEFAULT_BASE_URL = "https://api.getrouter.dev/codex"
ALIBABA_PROVIDER = "alibaba"
ALIBABA_MODEL = "qwen3.5-plus"
ALIBABA_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEEPSEEK_PROVIDER = "deepseek"
DEEPSEEK_MODEL = "deepseek-chat"


def _secure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(0o700)
    except OSError:
        pass


def _secure_file(path: Path) -> None:
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


def _load_codex_config(codex_home: Path) -> dict:
    config_path = codex_home / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Codex config not found: {config_path}")
    return tomllib.loads(config_path.read_text(encoding="utf-8"))


def _load_codex_auth(codex_home: Path) -> dict:
    auth_path = codex_home / "auth.json"
    if not auth_path.exists():
        raise FileNotFoundError(f"Codex auth not found: {auth_path}")
    return json.loads(auth_path.read_text(encoding="utf-8"))


def _load_existing_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _load_existing_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(
        yaml.safe_dump(data, allow_unicode=False, sort_keys=False),
        encoding="utf-8",
    )
    _secure_file(path)


def _wire_api_to_transport(wire_api: str) -> str:
    if (wire_api or "").strip().lower() == "responses":
        return "codex_responses"
    return "openai_chat"


def _build_provider_config(codex_config: dict, existing_env: dict[str, str]) -> tuple[str, dict, str | None]:
    provider_name = str(codex_config.get("model_provider") or DEFAULT_PROVIDER).strip() or DEFAULT_PROVIDER
    providers = codex_config.get("model_providers") or {}
    provider_cfg = providers.get(provider_name) if isinstance(providers, dict) else None
    provider_cfg = provider_cfg if isinstance(provider_cfg, dict) else {}

    base_url = str(provider_cfg.get("base_url") or DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
    transport = _wire_api_to_transport(str(provider_cfg.get("wire_api") or ""))

    return (
        provider_name,
        {
            "name": str(provider_cfg.get("name") or provider_name),
            "base_url": base_url,
            "key_env": "OPENAI_API_KEY",
            "transport": transport,
            "default_model": str(codex_config.get("model") or DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        },
        base_url,
    )


def _write_env(env_path: Path, api_key: str) -> None:
    managed: dict[str, str] = {"OPENAI_API_KEY": api_key}
    dashscope_key = str(os.getenv("DASHSCOPE_API_KEY") or "").strip()
    if dashscope_key:
        managed["DASHSCOPE_API_KEY"] = dashscope_key
        managed["DASHSCOPE_BASE_URL"] = str(os.getenv("DASHSCOPE_BASE_URL") or ALIBABA_BASE_URL).strip()
    deepseek_key = str(os.getenv("DEEPSEEK_API_KEY") or "").strip()
    if deepseek_key:
        managed["DEEPSEEK_API_KEY"] = deepseek_key

    lines: list[str] = []
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" not in line:
                lines.append(line)
                continue
            key = line.split("=", 1)[0]
            if key not in managed:
                lines.append(line)
    for key, value in managed.items():
        lines.append(f"{key}={value}")
    env_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    _secure_file(env_path)


def bootstrap(*, codex_home: Path, hermes_home: Path, repo_root: Path) -> None:
    codex_config = _load_codex_config(codex_home)
    codex_auth = _load_codex_auth(codex_home)
    existing_env = _load_existing_env(hermes_home / ".env")

    api_key = str(codex_auth.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(f"{codex_home / 'auth.json'} does not contain OPENAI_API_KEY")

    provider_name, provider_config, base_url = _build_provider_config(codex_config, existing_env)
    model_name = provider_config["default_model"]

    _secure_dir(hermes_home)
    for subdir in ("cron", "sessions", "logs", "memories"):
        _secure_dir(hermes_home / subdir)

    config_path = hermes_home / "config.yaml"
    env_path = hermes_home / ".env"
    config = _load_existing_yaml(config_path)

    model_cfg = config.get("model")
    if not isinstance(model_cfg, dict):
        model_cfg = {}
        config["model"] = model_cfg
    # Use the provider name from Codex config so Hermes resolves API key correctly.
    # The provider entry is always written to providers: for key_env resolution.
    model_cfg["provider"] = provider_name
    model_cfg["default"] = model_name
    if base_url:
        model_cfg["base_url"] = base_url
    else:
        model_cfg.pop("base_url", None)
    model_cfg["api_mode"] = provider_config["transport"]

    providers_cfg = config.get("providers")
    if not isinstance(providers_cfg, dict):
        providers_cfg = {}
        config["providers"] = providers_cfg
    providers_cfg[provider_name] = provider_config
    providers_cfg.setdefault(
        ALIBABA_PROVIDER,
        {
            "name": "Alibaba Cloud (DashScope)",
            "base_url": str(
                os.getenv("DASHSCOPE_BASE_URL")
                or existing_env.get("DASHSCOPE_BASE_URL")
                or ALIBABA_BASE_URL
            ).strip(),
            "key_env": "DASHSCOPE_API_KEY",
            "transport": "chat_completions",
            "default_model": ALIBABA_MODEL,
        },
    )

    deepseek_key = str(os.getenv("DEEPSEEK_API_KEY") or "").strip()
    if deepseek_key:
        providers_cfg.setdefault(
            DEEPSEEK_PROVIDER,
            {
                "name": "DeepSeek",
                "base_url": "https://api.deepseek.com/v1",
                "key_env": "DEEPSEEK_API_KEY",
                "transport": "chat_completions",
                "default_model": DEEPSEEK_MODEL,
            },
        )
        config["fallback_model"] = {
            "provider": DEEPSEEK_PROVIDER,
            "model": DEEPSEEK_MODEL,
        }

    terminal_cfg = config.get("terminal")
    if not isinstance(terminal_cfg, dict):
        terminal_cfg = {}
        config["terminal"] = terminal_cfg
    terminal_cfg.setdefault("backend", "local")
    terminal_cfg["cwd"] = str(repo_root)
    terminal_cfg.setdefault("timeout", 180)

    display_cfg = config.get("display")
    if not isinstance(display_cfg, dict):
        display_cfg = {}
        config["display"] = display_cfg
    display_cfg.setdefault("streaming", True)

    toolsets = config.get("toolsets")
    if not isinstance(toolsets, list) or not toolsets:
        config["toolsets"] = ["hermes-cli"]

    config.setdefault("_config_version", 18)
    _write_yaml(config_path, config)
    _write_env(env_path, api_key)

    print(f"HERMES_HOME={hermes_home}")
    print(f"Configured provider={provider_name} model={model_name}")
    print(f"Endpoint={base_url}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codex-home", default=os.getenv("CODEX_HOME", str(Path.home() / ".codex")))
    parser.add_argument("--hermes-home", default=os.getenv("HERMES_HOME", ""))
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()

    hermes_home = Path(args.hermes_home).expanduser().resolve() if args.hermes_home else Path(args.repo_root).resolve() / ".hermes-dev"
    bootstrap(
        codex_home=Path(args.codex_home).expanduser().resolve(),
        hermes_home=hermes_home,
        repo_root=Path(args.repo_root).expanduser().resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
