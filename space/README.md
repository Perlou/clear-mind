---
license: Apache License 2.0
tags:
  - chinese-llm
  - from-scratch
  - qwen3-compatible
  - clearmind
  - minimind-aligned
language:
  - zh
  - en
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
---

# 🧠 ClearMind Demo

ClearMind 是一个**从零训练**的中文小语言模型项目，覆盖 Pretrain → SFT → DPO 全流程，以 [MiniMind](https://github.com/jingyaogong/minimind) 为参照在同等规模上做架构升级与工程打磨。

## 🚧 当前状态

本 Demo 处于**全流程冒烟期**：base / plus 仍在训练中，sidebar 上展示的两个规格**当前都临时指向 small (15.68M) 占位**走通整条管线。等 base / plus 训完上架后，仅需更新两行 repo id（或设两个环境变量），无需改代码。

| 规格 | 目标参数量 | 当前指向 | 训完目标 |
|---|---|---|---|
| **ClearMind-Base** | 68.8M dense | `Perlou/ClearMind-Small`（占位） | `Perlou/ClearMind-Base` |
| **ClearMind-Plus** | 486.3M dense | `Perlou/ClearMind-Small`（占位） | `Perlou/ClearMind-Plus` |

> small (15.68M) 是项目内部冒烟规格，**不参与对外演示**，仅作为占位让 demo 管线在 base/plus 训完前就能上架。

## 功能

- 💬 多轮对话（可控历史轮次 0-8）
- 🤔 自适应思考模式（`<think>...</think>` 折叠展示）
- 🛠️ 工具调用（8 个 mock 工具：数学、时间、随机数、字数、单位、天气、汇率、翻译）
- 🌐 中英双语 UI
- ⚡ 流式输出（TextIteratorStreamer）
- 🎚️ 实时调节温度 / 最大长度 / 历史轮次

## 与 MiniMind 的关系

复用 MiniMind 的 tokenizer（vocab=6400，含 `<|im_start|>` / `<|im_end|>` / `<tool_call>` / `<think>`）与数据生态，在以下方面做了升级：

- **架构**：QK-Norm + RoPE θ=1e6 + YaRN 外推
- **训练工程**：BaseTrainer 抽象、val split + EarlyStopping、参数分组 weight decay、AMP 新 API、SWA、torch.compile 兼容 LoRA、fused AdamW、DDP no_sync、activation checkpointing
- **修复**：attention_mask `0*inf=NaN`、SFT loss-mask BPE 边界错位、RoPE buffer 共享、原子 checkpoint、RMSNorm bf16 dtype 一致性

## 配置（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `CLEARMIND_PLATFORM` | `ms` | `ms` 走 ModelScope SDK，`hf` 走 transformers |
| `CLEARMIND_REPO_BASE` | `Perlou/ClearMind-Small`（占位） | 训完改成 `Perlou/ClearMind-Base` |
| `CLEARMIND_REPO_PLUS` | `Perlou/ClearMind-Small`（占位） | 训完改成 `Perlou/ClearMind-Plus` |
| `CLEARMIND_USE_ZEROGPU` | `0` | HF Spaces 专用，MS 无效 |

## 致谢

- [MiniMind](https://github.com/jingyaogong/minimind) (jingyaogong)：tokenizer / 数据 / chat_template / demo 形态参考

## License

Apache 2.0
