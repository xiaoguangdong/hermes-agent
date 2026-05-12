#!/usr/bin/env python3
"""Serial oMLX benchmark runner for Hermes local evals.

Starts a fresh oMLX server for each config alias, waits for health, runs the
requested benchmark, then tears the server down before moving to the next
model. This avoids:

- 502s caused by local OpenAI-compatible traffic accidentally inheriting proxy
  settings from the environment.
- 507s caused by reusing a live oMLX server that still has a previous large
  model pinned in memory.
"""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OMLX_CLI = Path(
    os.environ.get("OMLX_CLI", "/Volumes/SPEED/Models/Olmx/.venv-omlx/bin/omlx")
)
DEFAULT_OMLX_BASE = Path(
    os.environ.get("OMLX_BASE_PATH", "/Volumes/SPEED/Models/Olmx/.olmx")
)
DEFAULT_OMLX_MODEL_DIR = Path(
    os.environ.get("OMLX_MODEL_DIR", "/Volumes/SPEED/Models/Olmx/models")
)
DEFAULT_HERMES_PYTHON = Path(
    os.environ.get("HERMES_PYTHON", str(REPO_ROOT / "venv/bin/python"))
)
DEFAULT_LOG_DIR = REPO_ROOT / "logs" / "omlx-local"

CONFIG_ALIASES = {
    "qwen36": REPO_ROOT / "environments/benchmarks/tblite/omlx-local.yaml",
    "supergemma26b": REPO_ROOT
    / "environments/benchmarks/tblite/omlx-local-supergemma26b.yaml",
}
BENCHMARK_ENTRYPOINTS = {
    "tblite": REPO_ROOT / "environments/benchmarks/tblite/tblite_env.py",
}


def _quote(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _listener_pids(port: int) -> list[int]:
    result = subprocess.run(
        ["lsof", "-tiTCP:%d" % port, "-sTCP:LISTEN"],
        capture_output=True,
        text=True,
        check=False,
    )
    pids: list[int] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pids.append(int(line))
        except ValueError:
            continue
    return pids


def _wait_for_exit(pid: int, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return True
        time.sleep(0.2)
    return False


def terminate_listeners(port: int, dry_run: bool = False) -> None:
    pids = _listener_pids(port)
    if not pids:
        return

    if dry_run:
        print(f"[dry-run] terminate listeners on :{port}: {pids}")
        return

    print(f"Stopping existing listeners on :{port}: {pids}")
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    for pid in pids:
        if _wait_for_exit(pid, timeout=10):
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def wait_for_health(host: str, port: int, proc: subprocess.Popen[str], timeout: int) -> None:
    url = f"http://{host}:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"oMLX exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except urllib.error.URLError:
            pass
        time.sleep(1)
    raise TimeoutError(f"Timed out waiting for healthy oMLX server at {url}")


def start_omlx_server(
    *,
    omlx_cli: Path,
    model_dir: Path,
    base_path: Path,
    host: str,
    port: int,
    log_dir: Path,
    dry_run: bool = False,
) -> tuple[subprocess.Popen[str] | None, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"omlx_{time.strftime('%Y%m%d_%H%M%S')}_{port}.log"
    cmd = [
        str(omlx_cli),
        "serve",
        "--model-dir",
        str(model_dir),
        "--base-path",
        str(base_path),
        "--port",
        str(port),
        "--host",
        host,
        "--no-cache",
        "--log-level",
        "info",
    ]
    if dry_run:
        print(f"[dry-run] start oMLX: {_quote(cmd)}")
        print(f"[dry-run] server log: {log_path}")
        return None, log_path

    print(f"Starting oMLX: {_quote(cmd)}")
    log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    wait_for_health(host, port, proc, timeout=60)
    print(f"oMLX healthy on http://{host}:{port} (pid={proc.pid}, log={log_path})")
    return proc, log_path


def stop_omlx_server(proc: subprocess.Popen[str] | None, port: int, dry_run: bool = False) -> None:
    if dry_run:
        print(f"[dry-run] stop oMLX on :{port}")
        return

    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    terminate_listeners(port)


def run_eval(
    *,
    hermes_python: Path,
    benchmark: str,
    config_path: Path,
    task_filter: str | None,
    extra_eval_args: Sequence[str],
    dry_run: bool = False,
) -> int:
    benchmark_entry = BENCHMARK_ENTRYPOINTS[benchmark]
    cmd = [str(hermes_python), str(benchmark_entry), "evaluate", "--config", str(config_path)]
    if task_filter:
        cmd.extend(["--env.task_filter", task_filter])
    cmd.extend(extra_eval_args)
    env = os.environ.copy()
    env.update(
        {
            "HF_ENDPOINT": env.get("HF_ENDPOINT", "https://hf-mirror.com"),
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "PYTHONUNBUFFERED": "1",
            "LOGLEVEL": env.get("LOGLEVEL", "INFO"),
            "NO_PROXY": env.get("NO_PROXY", "127.0.0.1,localhost"),
            "no_proxy": env.get("no_proxy", "127.0.0.1,localhost"),
        }
    )
    if dry_run:
        print(f"[dry-run] eval: {_quote(cmd)}")
        return 0

    print(f"Running eval: {_quote(cmd)}")
    completed = subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=False)
    return completed.returncode


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Hermes local oMLX benchmark configs serially."
    )
    parser.add_argument(
        "--config-alias",
        action="append",
        choices=sorted(CONFIG_ALIASES),
        dest="config_aliases",
        help="Config alias to run. Repeat to compare multiple models serially.",
    )
    parser.add_argument(
        "--benchmark",
        default="tblite",
        choices=sorted(BENCHMARK_ENTRYPOINTS),
        help="Benchmark entrypoint to use.",
    )
    parser.add_argument(
        "--task-filter",
        help="Comma-separated task names to run.",
    )
    parser.add_argument(
        "--extra-eval-arg",
        action="append",
        default=[],
        help="Extra arg forwarded to the benchmark command. Repeatable.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--omlx-cli", type=Path, default=DEFAULT_OMLX_CLI)
    parser.add_argument("--omlx-base-path", type=Path, default=DEFAULT_OMLX_BASE)
    parser.add_argument("--omlx-model-dir", type=Path, default=DEFAULT_OMLX_MODEL_DIR)
    parser.add_argument("--hermes-python", type=Path, default=DEFAULT_HERMES_PYTHON)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue to later configs even if one eval fails.",
    )
    return parser.parse_args(argv)


def validate_paths(paths: Iterable[Path]) -> None:
    for path in paths:
        if path.exists():
            continue
        raise FileNotFoundError(f"Missing required path: {path}")


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    aliases = args.config_aliases or ["qwen36"]
    config_paths = [CONFIG_ALIASES[alias] for alias in aliases]
    validate_paths(
        [
            args.omlx_cli,
            args.omlx_base_path,
            args.omlx_model_dir,
            args.hermes_python,
            *config_paths,
        ]
    )

    summaries: list[tuple[str, int]] = []
    for alias, config_path in zip(aliases, config_paths):
        print("")
        print("=" * 72)
        print(f"Model alias: {alias}")
        print(f"Config: {config_path}")
        print("=" * 72)
        terminate_listeners(args.port, dry_run=args.dry_run)
        proc = None
        try:
            proc, _ = start_omlx_server(
                omlx_cli=args.omlx_cli,
                model_dir=args.omlx_model_dir,
                base_path=args.omlx_base_path,
                host=args.host,
                port=args.port,
                log_dir=args.log_dir,
                dry_run=args.dry_run,
            )
            code = run_eval(
                hermes_python=args.hermes_python,
                benchmark=args.benchmark,
                config_path=config_path,
                task_filter=args.task_filter,
                extra_eval_args=args.extra_eval_arg,
                dry_run=args.dry_run,
            )
        finally:
            stop_omlx_server(proc, args.port, dry_run=args.dry_run)

        summaries.append((alias, code))
        if code != 0 and not args.keep_going:
            break

    print("")
    print("Summary:")
    for alias, code in summaries:
        status = "ok" if code == 0 else f"exit={code}"
        print(f"  {alias}: {status}")
    return 0 if all(code == 0 for _, code in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
