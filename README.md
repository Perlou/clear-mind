# 🧠 ClearMind

> 从零实现的中文 dense LLM。复用 [minimind](https://github.com/jingyaogong/minimind) 数据/tokenizer 生态，做架构升级与工程标准化，发布到 HuggingFace 和 ModelScope。

[![Demo](https://img.shields.io/badge/🎬_Demo-ModelScope-blue)](https://www.modelscope.cn/studios/Perlou/ClearMind-Demo)
[![Tests](https://img.shields.io/badge/tests-147%20passed-brightgreen)]()
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)]()

---

## 🎬 在线演示

**👉 [https://www.modelscope.cn/studios/Perlou/ClearMind-Demo](https://www.modelscope.cn/studios/Perlou/ClearMind-Demo)**

可选 Base / Plus，支持自适应思考（`<think>`）、工具调用、流式输出、中英双语 UI。

## 📦 模型仓库

| 规格 | 参数 | ModelScope（国内推荐） | HuggingFace |
|---|---|---|---|
| **ClearMind-Base** | **68.8M** dense | [Perlou/ClearMind-Base](https://www.modelscope.cn/models/Perlou/ClearMind-Base) | [Perlous/ClearMind-Base](https://huggingface.co/Perlous/ClearMind-Base) |
| **ClearMind-Plus** | **486.3M** dense | [Perlou/ClearMind-Plus](https://www.modelscope.cn/models/Perlou/ClearMind-Plus) | [Perlous/ClearMind-Plus](https://huggingface.co/Perlous/ClearMind-Plus) |

## 🚀 快速使用

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

repo = "Perlous/ClearMind-Base"   # 或 Perlous/ClearMind-Plus
tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, dtype=torch.float16)

messages = [{"role": "user", "content": "你好，介绍一下你自己"}]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors="pt", return_token_type_ids=False)
out = m.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

> 国内用户走 ModelScope：`from modelscope import AutoModelForCausalLM, AutoTokenizer` + `repo = "Perlou/ClearMind-Base"`

---

## 🎯 与 MiniMind 的关系

ClearMind 复用 MiniMind 的 tokenizer（vocab=6400，含 `<|im_start|>` / `<|im_end|>` / `<tool_call>` / `<think>`）和数据生态，**在同等规模追求效果反超**：

- **架构升级**：QK-Norm + RoPE θ=1e6 + YaRN 长上下文 + GQA + KV Cache + SDPA + 残差 1/√(2L) 缩放
- **工程标准化**：BaseTrainer 抽象 + val split + EarlyStopping + 原子 checkpoint + 参数分组 weight decay + torch.compile + fused AdamW + DDP no_sync + activation checkpointing
- **关键 bug 修复**：attention_mask `0*inf=NaN`、SFT loss-mask BPE 边界错位、RoPE buffer 共享、RMSNorm bf16 dtype 一致性、DPO `max_steps` 生效
- **完整发布管线**：safetensors + Qwen3 兼容导出 + HF/MS 双发 + OpenAI 兼容 API + Streamlit Demo + AutoDL 一键工具链

## 🏗️ 架构

```
Token IDs → Embedding → N × TransformerBlock → RMSNorm → LM Head → Logits
                          ├─ RMSNorm + QK-Norm → MHA/GQA + RoPE(θ=1e6, YaRN) + KV Cache + SDPA
                          └─ RMSNorm → SwiGLU FFN
```

发布时通过 `scripts/convert_to_qwen3.py` 把训练态 GPT 转成 `Qwen3ForCausalLM` 格式 → `transformers` / `vLLM` / `Ollama (GGUF)` 即用，键名映射 `w_q/w_k/w_v/w_o → q_proj/k_proj/v_proj/o_proj`。

## 📐 配置矩阵

| Config | 参数 | d_model / heads / kv / layers | d_ff | seq_len | 用途 |
|---|---|---|---|---|---|
| `tiny.yaml` | 0.5M | 64 / 4 / 2 / 2 | 256 | 128 | CPU/MPS 冒烟（5 min） |
| `small.yaml` | 26M | 512 / 8 / 2 / 8 | 1664 | 1024 | 单卡验证（30-60 min） |
| **`main.yaml`** | **68.8M** | 768 / 8 / 4 / 8 | 2432 | 1024 | **ClearMind-Base 发布版** |
| **`plus.yaml`** | **486.3M** | 1280 / 16 / 4 / 24 | 4032 | 1024 | **ClearMind-Plus 发布版** |

`d_ff = ⌈d_model · π / 64⌉ · 64`（minimind / Qwen3 风格 TensorCore 对齐）。

---

## 🛠️ 一键工作流

```bash
# 1) 环境
git clone <this-repo> && cd clear-mind
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2) 数据（从 minimind_dataset 镜像，国内推荐 modelscope）
python scripts/download_data.py --profile base --source modelscope

# 3) 本地冒烟（CPU/MPS 即可）
bash run.sh                        # 交互式
python scripts/smoke_test.py --clean   # 端到端 ~5 min

# 4) AutoDL 正式训练（断 SSH 不影响）
bash scripts/autodl/preflight.sh --profile base
bash scripts/autodl/launch.sh base all       # ~8h 全流程
bash scripts/autodl/launch.sh plus all       # ~32h 全流程

# 5) 评估
python evaluate/benchmarks/ceval.py --config configs/main.yaml
python evaluate/benchmarks/cmmlu.py --config configs/main.yaml
python evaluate/benchmarks/alignbench.py --config configs/main.yaml

# 6) 发布到 HF + ModelScope
bash scripts/release.sh base \
    --push-hf Perlous/ClearMind-Base \
    --push-ms Perlou/ClearMind-Base
```

## 📁 项目结构

```
clear-mind/
├── src/                    # 模型 / trainer / dataset / inference
├── scripts/
│   ├── train.py            # 统一训练入口（--stage pretrain/sft/dpo/distillation/grpo）
│   ├── release.sh          # ⭐ 端到端发布流水线
│   ├── autodl/             # ⭐ AutoDL 一键工具链（preflight/launch/status/save_outputs）
│   └── {convert_to_qwen3,push_to_hub,push_to_modelscope,smoke_test}.py
├── evaluate/               # C-Eval / CMMLU / AlignBench / LLM-as-Judge
├── deploy/                 # OpenAI 兼容 API + Web demo + Dockerfile
├── space/                  # ⭐ Streamlit demo（推到 ModelScope 创空间）
├── configs/                # tiny / small / main / plus
├── tests/                  # 147 个单元测试
└── docs/
    ├── RELEASE_GUIDE.md    # ⭐ 模型权重发布完整手册
    ├── DEMO_DEPLOY.md      # ⭐ Demo 部署到 MS 创空间
    ├── AUTODL_GUIDE.md     # AutoDL 训练上线
    ├── DEPLOY.md           # 推理服务部署
    └── PRD.md / TECHNICAL_DESIGN.md / PROGRESS_TRACKER.md
```

## 📚 详细文档

| 文档 | 内容 |
|---|---|
| [docs/AUTODL_GUIDE.md](docs/AUTODL_GUIDE.md) | AutoDL 训练上线（rent + 自检 + 训练 + 评估 + 归档） |
| [docs/RELEASE_GUIDE.md](docs/RELEASE_GUIDE.md) | 模型权重发布到 HF + ModelScope（端到端） |
| [docs/DEMO_DEPLOY.md](docs/DEMO_DEPLOY.md) | Demo 部署到 ModelScope 创空间（含 token 持久化） |
| [docs/DEPLOY.md](docs/DEPLOY.md) | 推理服务部署（OpenAI 兼容 API / Gradio / Docker） |
| [docs/TECHNICAL_DESIGN.md](docs/TECHNICAL_DESIGN.md) | 架构与各模块设计 |
| [docs/PRD.md](docs/PRD.md) | 产品需求与目标 |
| [docs/PROGRESS_TRACKER.md](docs/PROGRESS_TRACKER.md) | 阶段进度 |
| [evaluate/README.md](evaluate/README.md) | 评测体系 |
| [CLAUDE.md](CLAUDE.md) | Claude Code 协作的项目指南 |

## 🧪 测试

```bash
./venv/bin/python -m pytest tests/ -q                  # 147 passed
./venv/bin/python -m ruff check src/ scripts/ tests/   # lint 全绿
./venv/bin/python scripts/smoke_test.py --clean        # 端到端 ~5 min
```

## 🙏 致谢

- 数据 / tokenizer / chat_template：[jingyaogong/minimind](https://github.com/jingyaogong/minimind)（Apache-2.0）
- 架构灵感：Llama / Qwen / Gemma
- 算法：RoFormer (RoPE) / GQA / SwiGLU / DPO / GRPO / CISPO

## 📝 License

Apache-2.0（与 minimind 数据 / tokenizer 来源协议保持一致）。

---

> **声明**：本项目复用 minimind 的开源数据与 tokenizer（Apache-2.0），独立实现了模型架构、trainer、dataset 适配层、推理 / 评测 / 发布工具链与 Streamlit demo。所有 minimind 借鉴点在代码注释中均有标注。
