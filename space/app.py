"""
ClearMind Web Demo (Streamlit)
==============================

跟 MiniMind 形态保持一致的对话演示：
  - think 模式开关（自适应思考）
  - 8 个 mock 工具调用
  - 中英文 UI 双语切换
  - 多轮历史控制（0-8 轮）
  - 流式输出（TextIteratorStreamer）
  - sidebar 模型规格选择（small / base / plus）

本地跑：
  cd space
  streamlit run app.py

部署目标：
  - HuggingFace Spaces : https://huggingface.co/spaces/Perlous/ClearMind-Demo
  - ModelScope 创空间   : https://modelscope.cn/studios/Perlou/ClearMind-Demo

环境变量（可选）：
  CLEARMIND_REPO_BASE   覆盖 base repo id
  CLEARMIND_REPO_PLUS   覆盖 plus repo id
  CLEARMIND_PLATFORM    "hf" / "ms"，影响默认仓库 id（默认 hf）
  CLEARMIND_USE_ZEROGPU 设为 "1" 在 HF Spaces 上启用 ZeroGPU 装饰器
"""

import json
import os
import random
import re
from threading import Thread

import numpy as np
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# ---------- 平台与默认仓库 ----------
PLATFORM = os.environ.get("CLEARMIND_PLATFORM", "ms").lower()  # ms / hf

# 注意：HF 用户名 Perlous，ModelScope 用户名 Perlou
DEFAULT_REPOS = {
    "hf": {
        "ClearMind-Base": "Perlous/ClearMind-Base",
    },
    "ms": {
        "ClearMind-Base": "Perlou/ClearMind-Base",
    },
}

REPOS = DEFAULT_REPOS.get(PLATFORM, DEFAULT_REPOS["ms"]).copy()
# 允许通过环境变量覆盖单个规格的 repo id（部署期切换方便）
for k, env in [
    ("ClearMind-Base", "CLEARMIND_REPO_BASE"),
]:
    if os.environ.get(env):
        REPOS[k] = os.environ[env]

USE_ZEROGPU = os.environ.get("CLEARMIND_USE_ZEROGPU") == "1"

# ---------- 页面 ----------
st.set_page_config(page_title="ClearMind", initial_sidebar_state="expanded")

# MS 创空间在 iframe 顶部会注入 ~50-60px 平台 banner（标题/反馈/分享/全屏按钮等），
# 标题区直接被遮；本地或 HF Spaces 则没有此问题。
# 通过 CLEARMIND_TOP_PAD 显式覆盖；默认 ms=50, 其他=0
TOP_PAD_PX = int(os.environ.get("CLEARMIND_TOP_PAD", "50" if PLATFORM == "ms" else "0"))

st.markdown(
    """
    <style>
        .stButton button {
            border-radius: 50% !important;
            width: 32px !important;
            height: 32px !important;
            padding: 0 !important;
            background-color: transparent !important;
            border: 1px solid #ddd !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 14px !important;
            color: #666 !important;
            margin: 5px 10px 5px 0 !important;
        }
        .stButton button:hover {
            border-color: #999 !important;
            color: #333 !important;
            background-color: #f5f5f5 !important;
        }
        .stApp > div:last-child {
            margin-bottom: -35px !important;
        }
        .stBottom > div:first-child {
            bottom: 50px !important;
        }
    </style>
""",
    unsafe_allow_html=True,
)

# 顶部 padding（动态），覆盖默认 streamlit 的负 margin 行为
# 不强制 chat_input fixed（fixed 在 ModelScope iframe 嵌套下会破坏 streamlit 自身布局）
st.markdown(
    f"""
    <style>
        .stMainBlockContainer > div:first-child {{
            margin-top: -50px !important;
        }}
        /* 兼容 streamlit 1.32+ 新结构 */
        section.main > div:first-child,
        [data-testid="stAppViewContainer"] > .main > .block-container {{
            padding-top: 0px !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 多语言 ----------
LANG_TEXTS = {
    "zh": {
        "settings": "模型设定调整",
        "history_rounds": "历史对话轮次",
        "max_length": "最大生成长度",
        "temperature": "温度",
        "thinking": "思考",
        "tools": "工具",
        "language": "语言",
        "send": "给 ClearMind 发送消息",
        "disclaimer": "AI 生成内容可能存在错误，请仔细核实",
        "think_tip": "自适应思考；多轮对话或与 Tool Call 共存时可能不稳定",
        "tool_select": "工具选择（最多 4 个）",
        "model": "模型",
        "loading": "首次加载中（约 1-2 分钟）...",
    },
    "en": {
        "settings": "Model Settings",
        "history_rounds": "History Rounds",
        "max_length": "Max Length",
        "temperature": "Temperature",
        "thinking": "Thinking",
        "tools": "Tools",
        "language": "Language",
        "send": "Send a message to ClearMind",
        "disclaimer": "AI-generated content may be inaccurate, please verify",
        "think_tip": "Adaptive thinking; may be unstable with multi-turn or Tool Call",
        "tool_select": "Tool Selection (max 4)",
        "model": "Model",
        "loading": "Loading model (1-2 min on first run)...",
    },
}


def get_text(key: str) -> str:
    lang = st.session_state.get("lang", "zh")
    return LANG_TEXTS.get(lang, {}).get(key, LANG_TEXTS["zh"].get(key, key))


# ---------- 工具定义（与 minimind 完全一致） ----------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate_math",
            "description": "计算数学表达式",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前时间",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "default": "Asia/Shanghai"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "random_number",
            "description": "生成随机数",
            "parameters": {
                "type": "object",
                "properties": {"min": {"type": "integer"}, "max": {"type": "integer"}},
                "required": ["min", "max"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "text_length",
            "description": "计算文本长度",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "unit_converter",
            "description": "单位转换",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number"},
                    "from_unit": {"type": "string"},
                    "to_unit": {"type": "string"},
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "获取天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_exchange_rate",
            "description": "获取汇率",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_currency": {"type": "string"},
                    "to_currency": {"type": "string"},
                },
                "required": ["from_currency", "to_currency"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "translate_text",
            "description": "翻译文本",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "target_lang": {"type": "string"},
                },
                "required": ["text", "target_lang"],
            },
        },
    },
]

TOOL_SHORT_NAMES = {
    "calculate_math": "数学",
    "get_current_time": "时间",
    "random_number": "随机",
    "text_length": "字数",
    "unit_converter": "单位",
    "get_current_weather": "天气",
    "get_exchange_rate": "汇率",
    "translate_text": "翻译",
}


def execute_tool(tool_name: str, args: dict) -> dict:
    """Mock 工具实现 —— 仅作演示，真实集成请替换。"""
    import datetime

    try:
        if tool_name == "calculate_math":
            # 警告：eval 仅作 demo；生产环境请用 ast.literal_eval 或安全表达式解析器
            return {"result": eval(args.get("expression", "0"))}
        elif tool_name == "get_current_time":
            return {"result": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        elif tool_name == "random_number":
            return {"result": random.randint(args.get("min", 0), args.get("max", 100))}
        elif tool_name == "text_length":
            return {"result": len(args.get("text", ""))}
        elif tool_name == "unit_converter":
            return {
                "result": f"{args.get('value', 0)} {args.get('from_unit', '')} = ? {args.get('to_unit', '')} (mock)"
            }
        elif tool_name == "get_current_weather":
            return {"result": f"{args.get('city', 'Unknown')}: 晴, 7~10°C (mock)"}
        elif tool_name == "get_exchange_rate":
            return {
                "result": f"1 {args.get('from_currency', 'USD')} = 7.2 {args.get('to_currency', 'CNY')} (mock)"
            }
        elif tool_name == "translate_text":
            return {"result": f"[mock translation of: {args.get('text', '')[:30]}]"}
        return {"result": "Unknown tool"}
    except Exception as e:
        return {"error": str(e)}


# ---------- 思考块 / 工具调用块的 HTML 渲染 ----------
def process_assistant_content(content: str, is_streaming: bool = False) -> str:
    # ToolCall 块
    if "<tool_call>" in content:

        def fmt_tc(match):
            try:
                tc = json.loads(match.group(1))
                name = tc.get("name", "unknown")
                args = tc.get("arguments", {})
                return (
                    '<div style="background: rgba(80, 110, 150, 0.20); border: 1px solid rgba(140, 170, 210, 0.30); '
                    'padding: 10px 12px; border-radius: 12px; margin: 6px 0;">'
                    '<div style="font-size:12px;opacity:.75;display:block;margin:0 0 6px 0;line-height:1;">ToolCalling</div>'
                    f"<div><b>{name}</b>: {json.dumps(args, ensure_ascii=False)}</div></div>"
                )
            except Exception:
                return match.group(0)

        content = re.sub(
            r"<tool_call>(.*?)</tool_call>", fmt_tc, content, flags=re.DOTALL
        )

    # 流式生成 + open_thinking 早期，把整段当成"思考中"
    if (
        is_streaming
        and st.session_state.get("enable_thinking", False)
        and "</think>" not in content
        and "<think>" not in content
    ):
        m = re.search(r"(\n\n(?:我是|您好|你好)[^\n]*)", content)
        if m and m.start(1) > 5:
            i = m.start(1)
            think_part = content[:i]
            answer_part = content[i:]
            return (
                '<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;">'
                '<summary style="cursor: pointer; color: #888;">已思考</summary>'
                '<div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto;">'
                f"{think_part.strip()}</div></details>{answer_part}"
            )
        elif len(content) > 5:
            return (
                '<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;">'
                '<summary style="cursor: pointer; color: #888;">思考中...</summary>'
                '<div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto; '
                'display: flex; flex-direction: column-reverse;">'
                f'<div style="margin-bottom: auto;">{content.strip().replace(chr(10), "<br>")}</div></div></details>'
            )

    # <think>...</think> 完整匹配
    if "<think>" in content and "</think>" in content:

        def fmt_think(match):
            tc = match.group(2)
            if tc.replace("\n", "").strip():
                return (
                    '<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;">'
                    '<summary style="cursor: pointer; color: #888;">已思考</summary>'
                    '<div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto;">'
                    f"{tc.strip()}</div></details>"
                )
            return ""

        content = re.sub(
            r"(<think>)(.*?)(</think>)", fmt_think, content, flags=re.DOTALL
        )

    # <think> 开始但还没闭合
    if "<think>" in content and "</think>" not in content:

        def fmt_in_progress(match):
            tc = match.group(1)
            return (
                '<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;">'
                '<summary style="cursor: pointer; color: #888;">思考中...</summary>'
                '<div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto; '
                'display: flex; flex-direction: column-reverse;">'
                f'<div style="margin-bottom: auto;">{tc.strip().replace(chr(10), "<br>")}</div></div></details>'
            )

        content = re.sub(r"<think>(.*?)$", fmt_in_progress, content, flags=re.DOTALL)

    # 仅有 </think>
    if "<think>" not in content and "</think>" in content:

        def fmt_no_start(match):
            tc = match.group(1)
            if tc.replace("\n", "").strip():
                return (
                    '<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;">'
                    '<summary style="cursor: pointer; color: #888;">已思考</summary>'
                    '<div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto;">'
                    f"{tc.strip()}</div></details>"
                )
            return ""

        content = re.sub(r"(.*?)</think>", fmt_no_start, content, flags=re.DOTALL)

    return content


# ---------- 模型加载（cache） ----------
@st.cache_resource(show_spinner=True)
def load_model_tokenizer(repo_id: str):
    """从 HF / ModelScope 仓库加载 Qwen3 兼容模型。"""
    if PLATFORM == "ms":
        # ModelScope SDK 入口
        from modelscope import AutoModelForCausalLM as MSModel, AutoTokenizer as MSTok

        tokenizer = MSTok.from_pretrained(repo_id, trust_remote_code=True)
        model = MSModel.from_pretrained(repo_id, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(repo_id, trust_remote_code=True)

    if device == "cuda":
        model = model.half()
    model = model.eval().to(device)
    return model, tokenizer


# ---------- Sidebar ----------
selected_model = st.sidebar.selectbox(
    LANG_TEXTS["zh"]["model"],
    list(REPOS.keys()),
    index=0,  # Plus 排第一，默认选中
)
model_repo = REPOS[selected_model]

slogan_zh = f"我是 {selected_model}，有什么可以帮你的？"
slogan_en = f"I am {selected_model}, how can I help you?"
slogan = slogan_zh if st.session_state.get("lang", "zh") == "zh" else slogan_en

st.sidebar.markdown('<hr style="margin: 12px 0 16px 0;">', unsafe_allow_html=True)

# 语言
lang_options = {"中文": "zh", "English": "en"}
current_lang = st.session_state.get("lang", "zh")
lang_index = 0 if current_lang == "zh" else 1
lang_label = st.sidebar.radio(
    "Language / 语言", list(lang_options.keys()), index=lang_index, horizontal=True
)
if lang_options[lang_label] != current_lang:
    st.session_state.lang = lang_options[lang_label]
    st.rerun()

st.sidebar.markdown('<hr style="margin: 12px 0 16px 0;">', unsafe_allow_html=True)

# 参数
st.session_state.history_chat_num = st.sidebar.slider(
    get_text("history_rounds"), 0, 8, 0, step=2
)
st.session_state.max_new_tokens = st.sidebar.slider(
    get_text("max_length"), 128, 2048, 2048, step=64
)
st.session_state.temperature = st.sidebar.slider(
    get_text("temperature"), 0.6, 1.2, 0.90, step=0.01
)

st.sidebar.markdown('<hr style="margin: 12px 0 16px 0;">', unsafe_allow_html=True)

# 功能开关
st.session_state.enable_thinking = st.sidebar.checkbox(
    get_text("thinking"), value=False, help=get_text("think_tip")
)
st.session_state.selected_tools = []
with st.sidebar.expander(get_text("tools")):
    st.caption(get_text("tool_select"))
    selected_count = sum(
        1 for t in TOOLS if st.session_state.get(f"tool_{t['function']['name']}", False)
    )
    for tool in TOOLS:
        name = tool["function"]["name"]
        short = TOOL_SHORT_NAMES.get(name, name)
        checked = st.checkbox(
            short,
            key=f"tool_{name}",
            disabled=(
                selected_count >= 4 and not st.session_state.get(f"tool_{name}", False)
            ),
        )
        if checked and len(st.session_state.selected_tools) < 4:
            st.session_state.selected_tools.append(name)

# ---------- 标题 ----------
st.markdown(
    f'<div style="display: flex; flex-direction: column; align-items: center; text-align: center; margin: 0; padding: 0;">'
    '<div style="font-style: italic; font-weight: 900; margin: 0; padding-top: 4px; '
    'display: flex; align-items: center; justify-content: center; flex-wrap: wrap; width: 100%;">'
    f'<span style="font-size: 26px;">🧠 {slogan}</span></div>'
    f'<span style="color: #bbb; font-style: italic; margin-top: 6px; margin-bottom: 10px;">{get_text("disclaimer")}</span>'
    "</div>",
    unsafe_allow_html=True,
)


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ---------- ZeroGPU 装饰器（HF Spaces 上启用，可在 PLUS 上获得免费 A100） ----------
def maybe_zerogpu(func):
    """如果 USE_ZEROGPU=1 且 spaces 包可用，套上 @spaces.GPU 装饰器。"""
    if not USE_ZEROGPU:
        return func
    try:
        import spaces  # type: ignore

        return spaces.GPU(duration=60)(func)
    except ImportError:
        return func


@maybe_zerogpu
def stream_generate(
    model, tokenizer, prompt_ids, attention_mask, temperature, max_new_tokens
):
    """启动后台线程做 generate，主线程从 streamer 读 token。"""
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    generation_kwargs = dict(
        input_ids=prompt_ids,
        attention_mask=attention_mask,
        max_length=prompt_ids.shape[1] + max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.85,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )
    Thread(target=model.generate, kwargs=generation_kwargs, daemon=True).start()
    return streamer


# ---------- 主流程 ----------
def main():
    with st.spinner(get_text("loading")):
        model, tokenizer = load_model_tokenizer(model_repo)

    # 历史
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_messages = []

    messages = st.session_state.messages
    for m in messages:
        if m["role"] == "assistant":
            st.markdown(process_assistant_content(m["content"]), unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div style="display: flex; justify-content: flex-end;">'
                f'<div style="display: inline-block; margin: 10px 0; padding: 8px 12px; '
                f'background-color: #3d4450; border-radius: 22px; color: white;">{m["content"]}</div></div>',
                unsafe_allow_html=True,
            )

    prompt = st.chat_input(key="input", placeholder=get_text("send"))
    if not prompt:
        return

    # echo user message
    st.markdown(
        f'<div style="display: flex; justify-content: flex-end;">'
        f'<div style="display: inline-block; margin: 10px 0; padding: 8px 12px; '
        f'background-color: #3d4450; border-radius: 22px; color: white;">{prompt}</div></div>',
        unsafe_allow_html=True,
    )
    user_msg = {"role": "user", "content": prompt[-st.session_state.max_new_tokens :]}
    messages.append(user_msg)
    st.session_state.chat_messages.append(user_msg)

    placeholder = st.empty()
    setup_seed(random.randint(0, 2**32 - 1))

    # 工具 / system prompt
    tools = [
        t
        for t in TOOLS
        if t["function"]["name"] in st.session_state.get("selected_tools", [])
    ] or None
    sys_prompt = (
        []
        if tools
        else [
            {
                "role": "system",
                "content": "你是 ClearMind，一个从零训练的中文小语言模型。请用完整且友好的方式回答用户问题。",
            }
        ]
    )
    st.session_state.chat_messages = (
        sys_prompt
        + st.session_state.chat_messages[-(st.session_state.history_chat_num + 1) :]
    )

    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if st.session_state.get("enable_thinking", False):
        template_kwargs["open_thinking"] = True
    if tools:
        template_kwargs["tools"] = tools

    new_prompt = tokenizer.apply_chat_template(
        st.session_state.chat_messages, **template_kwargs
    )
    inputs = tokenizer(
        new_prompt, return_tensors="pt", truncation=True, return_token_type_ids=False
    ).to(device)

    # 第一轮生成
    streamer = stream_generate(
        model,
        tokenizer,
        inputs.input_ids,
        inputs.attention_mask,
        st.session_state.temperature,
        st.session_state.max_new_tokens,
    )
    answer = ""
    for new_text in streamer:
        answer += new_text
        placeholder.markdown(
            process_assistant_content(answer, is_streaming=True), unsafe_allow_html=True
        )

    # 工具调用循环（最多 16 轮，避免无限）
    full_answer = answer
    for _ in range(16):
        tool_calls = re.findall(r"<tool_call>(.*?)</tool_call>", answer, re.DOTALL)
        if not tool_calls:
            break
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
        tool_results_html = []
        for tc_str in tool_calls:
            try:
                tc = json.loads(tc_str.strip())
                result = execute_tool(tc.get("name", ""), tc.get("arguments", {}))
                st.session_state.chat_messages.append(
                    {"role": "tool", "content": json.dumps(result, ensure_ascii=False)}
                )
                tool_results_html.append(
                    '<div style="background: rgba(90, 130, 110, 0.20); border: 1px solid rgba(150, 200, 170, 0.30); '
                    'padding: 10px 12px; border-radius: 12px; margin: 6px 0;">'
                    '<div style="font-size:12px;opacity:.75;display:block;margin:0 0 6px 0;line-height:1;">ToolCalled</div>'
                    f"<div><b>{tc.get('name', '')}</b>: {json.dumps(result, ensure_ascii=False)}</div></div>"
                )
            except Exception:
                pass
        full_answer += "\n" + "\n".join(tool_results_html) + "\n"
        placeholder.markdown(
            process_assistant_content(full_answer, is_streaming=True),
            unsafe_allow_html=True,
        )

        # 把工具结果送回模型继续生成
        new_prompt = tokenizer.apply_chat_template(
            st.session_state.chat_messages, **template_kwargs
        )
        inputs = tokenizer(
            new_prompt,
            return_tensors="pt",
            truncation=True,
            return_token_type_ids=False,
        ).to(device)
        streamer = stream_generate(
            model,
            tokenizer,
            inputs.input_ids,
            inputs.attention_mask,
            st.session_state.temperature,
            st.session_state.max_new_tokens,
        )
        answer = ""
        for new_text in streamer:
            answer += new_text
            placeholder.markdown(
                process_assistant_content(full_answer + answer, is_streaming=True),
                unsafe_allow_html=True,
            )
        full_answer += answer

    answer = full_answer
    messages.append({"role": "assistant", "content": answer})
    st.session_state.chat_messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
