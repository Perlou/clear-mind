#!/usr/bin/env bash
# =========================================================
# launch.sh — AutoDL 训练启动器（断连生存版）
# =========================================================
#
# 核心保障：训练在 tmux session 里跑，SSH 断开 / 浏览器关闭 / WiFi 掉线 都不影响。
#
# 用法:
#   bash scripts/autodl/launch.sh <规格> <阶段> [extra_args...]
#
# 例：
#   bash scripts/autodl/launch.sh tiny  pretrain                # tiny 冒烟
#   bash scripts/autodl/launch.sh small all                     # small 全流程
#   bash scripts/autodl/launch.sh base  pretrain                # base 仅预训练
#   bash scripts/autodl/launch.sh base  sft                     # base 仅 SFT（自动加载 base/pretrain/final.pth）
#   bash scripts/autodl/launch.sh plus  all                     # plus 全流程
#
#   bash scripts/autodl/launch.sh base all --foreground         # 不进 tmux，直接前台跑（调试用）
#   bash scripts/autodl/launch.sh base all --kill               # 杀掉已有 session
#
# 断连后回来怎么办：
#   1) tmux ls                                  # 列出所有 session
#   2) tmux attach -t clearmind-base-all        # 重新接入（看实时日志）
#   3) Ctrl+b d                                 # 再次脱离（不杀进程）
#
# 训练状态查询（不进 tmux）：
#   bash scripts/autodl/status.sh
#
# =========================================================

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'
CYAN=$'\033[0;36m'; BOLD=$'\033[1m'; DIM=$'\033[2m'; NC=$'\033[0m'

# ---- 参数 ----
SCALE="${1:-}"
STAGE="${2:-all}"
shift 2 2>/dev/null || true

FOREGROUND=0
KILL_EXISTING=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --foreground) FOREGROUND=1; shift ;;
        --kill)       KILL_EXISTING=1; shift ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$SCALE" ]]; then
    head -30 "$0" | grep '^#' | sed 's/^# \{0,1\}//'
    exit 1
fi

case "$SCALE" in
    tiny)  CONFIG="configs/tiny.yaml" ;;
    small) CONFIG="configs/small.yaml" ;;
    base)  CONFIG="configs/main.yaml" ;;
    plus)  CONFIG="configs/plus.yaml" ;;
    *) echo "${RED}❌ 未知规格: $SCALE${NC}"; exit 1 ;;
esac

case "$STAGE" in
    all|pretrain|sft|dpo) ;;
    *) echo "${RED}❌ 未知阶段: $STAGE${NC}"; exit 1 ;;
esac

SESSION="clearmind-${SCALE}-${STAGE}"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/${SESSION}.log"
PID_FILE="${LOG_DIR}/${SESSION}.pid"
STATE_FILE="${LOG_DIR}/${SESSION}.state"

mkdir -p "$LOG_DIR"

# ---- --kill：先杀旧 session ----
if [[ "$KILL_EXISTING" -eq 1 ]]; then
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        tmux kill-session -t "$SESSION"
        echo -e "${YELLOW}已杀掉 tmux session: $SESSION${NC}"
    fi
    if [[ -f "$PID_FILE" ]]; then
        OLD_PID=$(cat "$PID_FILE")
        if kill -0 "$OLD_PID" 2>/dev/null; then
            kill "$OLD_PID"
            echo -e "${YELLOW}已杀掉进程: $OLD_PID${NC}"
        fi
        rm -f "$PID_FILE"
    fi
    exit 0
fi

# ---- 检查是否已有 session 在跑 ----
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  已有 session: $SESSION${NC}"
    echo -e "${DIM}    重连：tmux attach -t $SESSION${NC}"
    echo -e "${DIM}    杀掉：bash scripts/autodl/launch.sh $SCALE $STAGE --kill${NC}"
    exit 1
fi

# ---- Python 解释器 ----
PYTHON="${PYTHON:-./venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="python"

# ---- 构造各阶段训练命令 ----
build_cmd() {
    local stage="$1"
    local resume_arg=""
    case "$stage" in
        sft)
            if [[ -f "outputs/pretrain/final.pth" ]]; then
                resume_arg="--resume outputs/pretrain/final.pth"
            fi
            ;;
        dpo)
            if [[ -f "outputs/sft/final.pth" ]]; then
                resume_arg="--resume outputs/sft/final.pth"
            fi
            ;;
    esac
    echo "$PYTHON scripts/train.py --stage $stage --config $CONFIG $resume_arg ${EXTRA_ARGS[*]:-}"
}

# ---- 生成临时执行脚本（彻底绕开引号嵌套问题） ----
TMPSCRIPT="${LOG_DIR}/${SESSION}.run.sh"

# 脚本头：固定内容，用 'HEREDOC' 防止变量展开
cat > "$TMPSCRIPT" << 'HEREDOC'
#!/usr/bin/env bash
set -euo pipefail

_on_err() {
    echo "failed" > "$STATE_FILE"
    echo "训练异常退出: $(date '+%F %T')"
}
trap _on_err ERR
HEREDOC

# 脚本体：需要变量展开，去掉引号
cat >> "$TMPSCRIPT" << HEREDOC
STATE_FILE='$STATE_FILE'
PID_FILE='$PID_FILE'
LOG_FILE='$LOG_FILE'

cd '$PROJECT_DIR'

echo '============================================================'
echo "  规格 : $SCALE  ($CONFIG)"
echo "  阶段 : $STAGE"
echo "  开始 : \$(date '+%F %T')"
echo "  PID  : \$\$"
echo '============================================================'

echo \$\$ > '$PID_FILE'
echo 'running' > '$STATE_FILE'

HEREDOC

# 根据阶段追加训练命令
if [[ "$STAGE" == "all" ]]; then
    PRETRAIN_CMD=$(build_cmd pretrain)
    SFT_CMD=$(build_cmd sft)
    DPO_CMD=$(build_cmd dpo)
    cat >> "$TMPSCRIPT" << HEREDOC
echo
echo '>>> [1/3] Pretrain'
$PRETRAIN_CMD

echo
echo '>>> [2/3] SFT'
$SFT_CMD

echo
echo '>>> [3/3] DPO'
$DPO_CMD

HEREDOC
else
    SINGLE_CMD=$(build_cmd "$STAGE")
    cat >> "$TMPSCRIPT" << HEREDOC
echo
echo '>>> $STAGE'
$SINGLE_CMD

HEREDOC
fi

# 完成收尾
cat >> "$TMPSCRIPT" << HEREDOC
echo 'done' > '$STATE_FILE'
echo
echo "== 训练正常完成 == \$(date '+%F %T')"
HEREDOC

# 自动评估（仅 all/dpo）
if [[ "$STAGE" == "all" ]] || [[ "$STAGE" == "dpo" ]]; then
    cat >> "$TMPSCRIPT" << HEREDOC
echo
echo '== 自动评估 =='
$PYTHON evaluate/eval_perplexity.py --config $CONFIG --compare \
    2>&1 | tee -a '${LOG_FILE}.eval' || true
HEREDOC
fi

chmod +x "$TMPSCRIPT"

# ---- 写入 banner 到 log ----
{
    echo "============================================================"
    echo "ClearMind 训练 session: $SESSION"
    echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  规格     : $SCALE ($CONFIG)"
    echo "  阶段     : $STAGE"
    echo "  Log      : $LOG_FILE"
    echo "  PID file : $PID_FILE"
    echo "  State    : $STATE_FILE"
    echo "============================================================"
} > "$LOG_FILE"

# ---- 模式 1：foreground（前台调试用） ----
if [[ "$FOREGROUND" -eq 1 ]]; then
    echo -e "${CYAN}前台模式（断 SSH 会被杀！）${NC}"
    bash "$TMPSCRIPT" 2>&1 | tee -a "$LOG_FILE"
    exit "${PIPESTATUS[0]}"
fi

# ---- 模式 2：tmux（默认，断连保护） ----
if ! command -v tmux &>/dev/null; then
    echo -e "${RED}❌ tmux 未安装；改用 nohup 兜底${NC}"
    nohup bash "$TMPSCRIPT" >> "$LOG_FILE" 2>&1 &
    NOHUP_PID=$!
    echo "$NOHUP_PID" > "$PID_FILE"
    disown
    echo -e "${GREEN}已用 nohup 启动，PID=$NOHUP_PID${NC}"
    echo -e "${DIM}查看日志: tail -f $LOG_FILE${NC}"
    exit 0
fi

# tmux 直接执行脚本文件，不再有任何引号嵌套
tmux new-session -d -s "$SESSION" \
    "bash '$TMPSCRIPT' 2>&1 | tee -a '$LOG_FILE'"

# 等 1s 让 tmux 启动稳定
sleep 1

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo -e "${GREEN}${BOLD}✅ 训练已在 tmux 里启动${NC}"
    echo ""
    echo -e "  ${BOLD}Session :${NC} $SESSION"
    echo -e "  ${BOLD}脚本    :${NC} $TMPSCRIPT"
    echo -e "  ${BOLD}日志    :${NC} $LOG_FILE"
    echo -e "  ${BOLD}PID file:${NC} $PID_FILE"
    echo ""
    echo -e "${CYAN}操作:${NC}"
    echo -e "  ${DIM}# 实时跟踪日志（不影响训练）${NC}"
    echo -e "  tail -f $LOG_FILE"
    echo ""
    echo -e "  ${DIM}# 重连 tmux 看实时画面${NC}"
    echo -e "  tmux attach -t $SESSION"
    echo -e "  ${DIM}（脱离：按 Ctrl+b d）${NC}"
    echo ""
    echo -e "  ${DIM}# 查询当前训练状态${NC}"
    echo -e "  bash scripts/autodl/status.sh"
    echo ""
    echo -e "  ${DIM}# 强制中止训练${NC}"
    echo -e "  bash scripts/autodl/launch.sh $SCALE $STAGE --kill"
    echo ""
    echo -e "${YELLOW}💡 提示：${NC}现在可以放心断开 SSH，训练会继续。"
    echo -e "${YELLOW}   即使 tmux 意外死掉，日志 $LOG_FILE 也会保留，${NC}"
    echo -e "${YELLOW}   且 outputs/<stage>/_resume.pth 会让你的训练自动续训。${NC}"
else
    echo -e "${RED}❌ tmux session 启动失败，请查看 $LOG_FILE${NC}"
    echo -e "${DIM}生成的临时脚本内容如下：${NC}"
    cat "$TMPSCRIPT"
    exit 1
fi