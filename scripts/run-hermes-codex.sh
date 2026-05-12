#!/usr/bin/env bash
set -euo pipefail

# Resolve symlinks so the script still finds the repository when invoked via
# ~/.local/bin/hermes.
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do
  DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
  SOURCE="$(readlink "$SOURCE")"
  [[ "$SOURCE" != /* ]] && SOURCE="$DIR/$SOURCE"
done
REPO_ROOT="$(cd -P "$(dirname "$SOURCE")/.." && pwd)"
export HERMES_HOME="${HERMES_HOME:-$REPO_ROOT/.hermes-dev}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO_ROOT/.uv-cache}"
export HERMES_CODEX_BOOTSTRAPPED=1
CONDA_PKGS_ROOT="${CONDA_PREFIX:-/opt/homebrew/Caskroom/miniconda/base}/pkgs"
CONDA_SITE_PACKAGES="${CONDA_PREFIX:-/opt/homebrew/Caskroom/miniconda/base}/lib/python3.13/site-packages"
unset PYTHONPATH

cd "$REPO_ROOT"

if [ ! -d "venv" ]; then
  python3 -m venv venv
fi

source venv/bin/activate

if ! python -c "import openai, yaml, rich, prompt_toolkit" >/dev/null 2>&1; then
  python -m pip install --no-build-isolation -e ".[dev,pty]"
fi

# Only fall back to bundled Conda prompt-toolkit bits when the venv still
# lacks the interactive TUI stack. Do not prepend the whole Conda site-packages
# tree, or we risk mixing incompatible runtime deps like openai/pydantic.
if ! python -c "import prompt_toolkit, wcwidth" >/dev/null 2>&1; then
  for pkg in prompt-toolkit* wcwidth* jedi* parso*; do
    for site_dir in "$CONDA_PKGS_ROOT"/$pkg/site-packages; do
      if [ -d "$site_dir" ]; then
        export PYTHONPATH="$site_dir${PYTHONPATH:+:$PYTHONPATH}"
      fi
    done
  done
fi

if [ ! -e "venv/bin/hermes" ]; then
  ln -sf "$REPO_ROOT/hermes" "venv/bin/hermes"
fi

python - <<'PY' >/tmp/hermes_python_env.log 2>&1
import json, os, sys
print("PYTHONPATH=", os.getenv("PYTHONPATH"))
print("prefix=", sys.prefix)
print("base_prefix=", sys.base_prefix)
print(json.dumps(sys.path, ensure_ascii=False, indent=2))
PY

python scripts/bootstrap_codex_hermes.py --hermes-home "$HERMES_HOME" --repo-root "$REPO_ROOT" >/dev/null

if [ -f "$HERMES_HOME/.env" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$HERMES_HOME/.env"
  set +a
fi

exec python -m hermes_cli.main "$@"
