#!/usr/bin/env bash
# =========================================================================
# with_env.sh — 在执行命令前自动加载项目根 .env
# =========================================================================
#
# 用途：让 .env 里的环境变量（如 MODELSCOPE_API_TOKEN）对接下来要跑的
# 命令可见，而不必污染你的 shell 全局环境，也不需要在 push_to_modelscope.py
# 里改代码。
#
# 用法：
#   bash scripts/with_env.sh ./venv/bin/python scripts/push_to_modelscope.py \
#       --model_dir release/clearmind-base --repo Perlou/ClearMind-Base
#
# 等价于：
#   set -a; source .env; set +a
#   ./venv/bin/python scripts/push_to_modelscope.py ...
#
# 但前者不会污染当前 shell（变量只对子进程可见）。
# =========================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "❌ 未找到 ${ENV_FILE}" >&2
    echo "   请先 cp .env.example .env 并填入真实 token" >&2
    exit 1
fi

# 关键：set -a 让接下来 source 的所有变量自动 export 到环境
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

if [[ $# -eq 0 ]]; then
    echo "✅ .env 已加载，但未提供要执行的命令" >&2
    echo "   用法: bash scripts/with_env.sh <command> [args...]" >&2
    exit 1
fi

exec "$@"
