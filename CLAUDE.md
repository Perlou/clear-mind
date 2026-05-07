# CLAUDE.md — ClearMind 项目指南

## 项目概述

ClearMind 是一个从零构建的中文 LLM 训练项目，覆盖 Pretrain → SFT → DPO（→ 计划中的蒸馏 / PPO / GRPO / Agentic RL）全流程，**目标是发布到 HuggingFace 与 ModelScope，并在同规模上反超 [minimind](https://github.com/jingyaogong/minimind)**。

ClearMind 与 minimind 的关系：复用 minimind 的 tokenizer 与数据生态（chat_template / tool_call / `<think>`），保留更扎实的工程基础（BaseTrainer 抽象、val split + EarlyStopping、参数分组 weight decay、跨平台设备、AMP 新 API、SWA、与 torch.compile 兼容的 LoRA），并通过架构升级（残差初始化 1/√(2L) 缩放、QK-Norm、RoPE θ=1e6、YaRN、修复已知 bug）追求超越。

## 双发布矩阵

| 产品 | 配置 | 参数量 | 对标 |
|---|---|---|---|
| **ClearMind-Base** | `configs/main.yaml` | **68.8M** dense | minimind-3 (64M dense) |
| **ClearMind-Plus** | `configs/plus.yaml` | **486.3M** dense | minimind-3-moe (198M-A64M)，单 token 算力 7.1× |

## 架构

- **模型**: GPT (RoPE + RMSNorm + SwiGLU + GQA + KV Cache)，发布时通过转换脚本对齐 `Qwen3ForCausalLM`
- **训练**: PreTrainer / SFTTrainer / DPOTrainer，继承自 BaseTrainer
- **推理**: KV Cache 自回归生成，Top-k/Top-p 采样
- **Tokenizer**: 默认走 HuggingFace tokenizer（minimind 的 ByteLevel BPE，vocab=6400，含 `<|im_start|>` / `<|im_end|>` / `<tool_call>` / `<think>` 与 16 个 buffer token），sentencepiece 路径作为 legacy 保留

## 关键路径

| 模块 | 路径 |
|------|------|
| 模型定义 | `src/model/` (config.py, gpt.py, attention.py, rope.py, transformer.py, feedforward.py, normalization.py) |
| 训练逻辑 | `src/training/` (base_trainer.py, pretrain.py, sft.py, dpo.py, lora.py) |
| 数据处理 | `src/data/` (hf_tokenizer.py, pretrain_dataset.py, sft_dataset.py, dpo_dataset.py, rl_dataset.py, tokenizer.py[legacy]) |
| 推理 | `src/inference/` (generate.py, chat.py) |
| 入口脚本 | `scripts/train.py` (统一训练入口，`--stage pretrain/sft/dpo`) |
| 配置文件 | `configs/{tiny,small,main,plus}.yaml`（与 ModelConfig.tiny()/.small()/.main()/.plus() 对齐） |
| 测试 | `tests/` |
| **Tokenizer 资产** | **`tokenizer/minimind/`**（顶层目录，git 追踪；从 minimind 复制） |
| 训练数据 | `data/`（gitignore，不上传仓库；用户从 minimind_dataset 自行下载） |

## 数据集（与 minimind 一致的扁平结构）

数据**不上传到 GitHub**（已 gitignore），用户从 [modelscope/gongjy/minimind_dataset](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) 或 [huggingface/jingyaogong/minimind_dataset](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main) 自行下载，放入 `data/`：

```
data/                              # gitignore
├── pretrain_t2t_mini.jsonl       # 预训练，每行 {"text": "..."}
├── sft_t2t_mini.jsonl            # SFT，每行 {"conversations": [...]}
├── dpo.jsonl                     # DPO，每行 {"chosen":[...], "rejected":[...]}
├── rlaif.jsonl                   # PPO/GRPO（计划中）
└── agent_rl.jsonl                # Agentic RL（计划中）

tokenizer/minimind/                # 仓库自带，git 追踪
├── tokenizer.json
└── tokenizer_config.json
```

`scripts/train.py` 的 STAGE_DEFAULTS 默认指向 `data/<filename>.jsonl`，无需 `--data` 显式覆盖。

## 配置文件规格

| 配置 | 参数量 | d_model / heads / kv_heads / layers | d_ff | max_seq_len | 适用 |
|---|---|---|---|---|---|
| `configs/tiny.yaml` | ~0.5M | 64 / 4 / 2 / 2 | 256 | 128 | CPU/MPS 冒烟 |
| `configs/small.yaml` | ~26M | 512 / 8 / 2 / 8 | 1664 | 1024 | 单卡真训（对齐 minimind2-small） |
| `configs/main.yaml` | **68.8M** | 768 / 8 / 4 / 8 | 2432 | 1024 | **ClearMind-Base 发布版** |
| `configs/plus.yaml` | **486.3M** | 1280 / 16 / 4 / 24 | 4032 | 1024 | **ClearMind-Plus 发布版** |

`d_ff = ⌈d_model · π / 64⌉ · 64`（minimind / Qwen3 风格，TensorCore/SIMD 对齐）。

## 开发命令

```bash
# 运行测试（147 个，CPU 上 ~3 秒）
./venv/bin/python -m pytest tests/ -q

# Lint（修自动可修部分；剩余项已加 noqa 镇压）
./venv/bin/python -m ruff check src/ scripts/ tests/
./venv/bin/python -m ruff check src/ scripts/ tests/ --fix

# 端到端冒烟（CPU/MPS 上 5–10 分钟，跑 tiny pretrain → SFT → chat 推理整链路）
./venv/bin/python scripts/smoke_test.py --clean

# 一键交互式训练（推荐入门）
bash run.sh

# 训练（默认走 HF tokenizer + minimind 数据）
./venv/bin/python scripts/train.py --stage pretrain --config configs/tiny.yaml
./venv/bin/python scripts/train.py --stage sft      --config configs/main.yaml
./venv/bin/python scripts/train.py --stage dpo      --config configs/plus.yaml

# 自定义 tokenizer（默认 tokenizer/minimind）
./venv/bin/python scripts/train.py --stage pretrain --config configs/main.yaml \
    --tokenizer /path/to/another/hf-tokenizer-dir

# AutoDL 上线（断 SSH 不影响）
bash scripts/autodl/preflight.sh --profile base   # 9 项强制自检
bash scripts/autodl/launch.sh    base all         # tmux 内全流程
bash scripts/autodl/status.sh                     # 状态查询
bash scripts/autodl/save_outputs.sh base          # 归档下载
```

## 已知 bug 修复 / 状态

| Bug / 改进 | 状态 |
|---|---|
| `GPT.forward` attention_mask `0*inf=NaN`（影响所有 SFT/DPO） | ✅ 已修复（`src/model/gpt.py:185-201`） |
| SFT loss-mask BPE 边界错位（旧实现用子串截取） | ✅ 已修复（HFTokenizer.generate_assistant_mask 用 token 序列扫描） |
| `_optimizer_step` LR 错位（scaler 跳过更新仍 scheduler.step） | ✅ 已修复 |
| 每层重复 RoPE buffer | ✅ 已修复（顶层共享，`src/model/gpt.py:82-83`） |
| 冗余 causal_mask buffer | ✅ 已修复（按需动态构造） |
| DPO chosen/rejected 两次独立 forward + deepcopy 创建 ref | ✅ 已修复（单 forward + 共享 ref） |
| 残差 proj 初始化未做 1/√(2L) 缩放 | ✅ 已修复 |
| DataLoader 默认 num_workers=0 / 无 persistent_workers | ✅ 已修复（智能默认） |
| Checkpoint 非原子保存 + 全 fp32 落盘 | ✅ 已修复（atomic write + half_weights） |
| **DPOTrainer 忽略 `config.max_steps`**（yaml 限制无效） | ✅ 已修复 2026-05-02（`src/training/dpo.py:81-89`，对齐 sft.py 写法） |
| **RMSNorm bf16 输出退化为 fp32**（破坏 autocast dtype 一致性） | ✅ 已修复 2026-05-02（`src/model/normalization.py`，内部 fp32 计算 + `.type_as(x)` 回原 dtype） |
| **DPO `_compute_log_probs_pair` 末尾 unreachable return** | ✅ 已修复 2026-05-07（`src/training/dpo.py`，删除死代码） |
| **`config.py __main__` 引用不存在的 `ModelConfig.medium()`** | ✅ 已修复 2026-05-07（改成 tiny/small/main/plus 四档） |
| **`download_data.py` 末尾推荐已废弃的 `scripts/autodl_train.sh`** | ✅ 已修复 2026-05-07（改推荐 `scripts/autodl/preflight.sh + launch.sh`） |
| **`requirements.txt` 缺 pytest/ruff/tensorboard，且 transformers 版本未封顶** | ✅ 已修复 2026-05-07（锁 `transformers>=4.40,<5` + 加 dev deps，避免 5.x 与 4.x 行为分叉） |
| ruff 51 项 lint 错（unused import / 空 f-string / E402 / E731 / F841） | ✅ 已修复 2026-05-07（38 项自动 fix + 13 项 `# noqa` 镇压，零功能改动） |

## 代码规范

- Python 3.10+，使用 Ruff 做 lint/format
- 中英文双语注释，docstring 解释"为什么"
- 模型属性名: `w_q`, `w_k`, `w_v`, `w_o`（保留 from-scratch 风格；发布时通过 convert 脚本映射成 `q_proj/k_proj/...` 对齐 Qwen3）
- Trainer 通过 BaseTrainer 共享逻辑，子类只实现 `train()`
- 配置通过 YAML + dataclass (ModelConfig)，工厂方法 `.tiny()/.small()/.main()/.plus()` 默认值与 yaml 完全对齐
- 数据集文件名不带子目录前缀，与 minimind 发布格式一致

## 测试

147 个测试覆盖所有核心模块（`pytest tests/`）。其中：
- 14 个 dataset 测试（`tests/test_datasets.py`）覆盖 PretrainDataset packed/per_sample、SFTDataset conversations/Alpaca/loss-mask、DPODataset conversations/字符串 fallback/loss-mask
- 5 个 RMSNorm dtype 测试（`tests/test_normalization.py`）覆盖 bf16/fp16 内部 fp32 计算契约
- DPO max_steps override 回归测试（`tests/test_training_edge_cases.py::test_dpo_trainer_respects_config_max_steps`）
- 其它测试不依赖 transformers，可在最小环境跑
- `conftest.py` 提供 `hf_tokenizer` fixture（session 作用域，缺 transformers 自动 skip）

测试不需要 GPU，全部在 CPU 上运行。

## 依赖

主要依赖（见 `requirements.txt`）：
- 核心：`torch>=2.1`, `transformers>=4.40,<5`（**封顶 5.0**：5.x 在 chat_template / from_pretrained 序列化等处不向后兼容；本地与 AutoDL 必须对齐到 4.x）, `safetensors`, `jinja2`
- Legacy 路径：`sentencepiece`（与旧 ClearMindTokenizer 共存，但默认不使用）
- 数据：`datasets`, `pyyaml`, `numpy`, `tqdm`, `modelscope`, `huggingface_hub`
- 训练监控：`tensorboard`（main/plus.yaml 默认 `use_tensorboard: true`）
- Dev / preflight：`pytest`, `ruff`（preflight.sh 第 8 步会跑 pytest，缺这两个 preflight 直接报 ERROR）

虚拟环境：`./venv/`（Python 3.10+）。**新克隆后第一步**：`./venv/bin/pip install -r requirements.txt`，否则 preflight 会因为缺 pytest 阻塞。

## 路线图（与 minimind 对齐 / 超越路径）

参考 `~/.claude/plans/cryptic-tumbling-dove.md` 与 `docs/PRD.md`：

- ✅ **Phase 1**（地基）：bug 修复 + chat_template + DPO 单 forward + RoPE buffer 共享 + 残差初始化 + DataLoader 默认 + 原子 checkpoint
- ✅ **Phase 2**（架构升级）：QK-Norm + RoPE θ=1e6 + YaRN + d_ff 对齐
- 🟢 **Phase 3**（训练阶段扩展）：Pretrain / SFT / DPO / 白盒蒸馏 / GRPO+CISPO / Rollout 引擎（Torch + SGLang 双 backend）已就绪；PPO / Agentic RL 列入 Phase 3.x 待补
- ✅ **Phase 4**（工程化）：torch.compile + wandb/swanlab + SkipBatchSampler + **fused AdamW** + **DDP no_sync** + **activation checkpointing**（main/plus.yaml 已默认开启推荐项）
- ✅ **Phase 5**（发布）：OpenAI 兼容 API server + Qwen3 兼容导出（`convert_to_qwen3.py`）+ safetensors + HF/ModelScope push + **`scripts/release.sh` 端到端流水线**（含 transformers 加载验证）
- ✅ **Phase E1**（评测）：C-Eval / CMMLU / AlignBench-zh + LLM-as-Judge + 多模型对照（`evaluate/benchmarks/` + `evaluate/judge/` + `eval_compare.py`）
- ✅ **AutoDL 上线**：preflight + tmux launch + status + save_outputs + release 全套脚本，断 SSH 不影响
