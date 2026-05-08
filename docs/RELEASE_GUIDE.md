# 📦 ClearMind 发布指南（HuggingFace + ModelScope）

> 把训练好的 `outputs/dpo/final.pth` 发布到 HuggingFace 与 ModelScope 双仓库的端到端手册。
>
> 本文件聚焦**模型权重对外发布**（即"上架"）。如果你要做的是推理服务化（OpenAI 兼容 API、Gradio、Docker），看 [`DEPLOY.md`](DEPLOY.md)。如果你要做的是训练上线（AutoDL preflight / launch / status / save），看 [`AUTODL_GUIDE.md`](AUTODL_GUIDE.md)。

---

## 📋 目录

1. [发布是什么 / 不是什么](#发布是什么--不是什么)
2. [核心概念：训练态 vs 发布态](#核心概念训练态-vs-发布态)
3. [Phase 1 — 准备工作](#phase-1--准备工作)
4. [Phase 2 — 转 Qwen3 兼容格式](#phase-2--转-qwen3-兼容格式)
5. [Phase 3 — 本地加载验证](#phase-3--本地加载验证)
6. [Phase 4 — 推送到 Hub](#phase-4--推送到-hub)
7. [Phase 5 — 远程圆环验证](#phase-5--远程圆环验证)
8. [三档完整命令对照（small / base / plus）](#三档完整命令对照small--base--plus)
9. [一键流水线：`scripts/release.sh`](#一键流水线scriptsreleasesh)
10. [README.md 模型卡模板](#readmemd-模型卡模板)
11. [故障排查 FAQ](#故障排查-faq)
12. [附录 A：版本对齐矩阵](#附录-a版本对齐矩阵)
13. [附录 B：文件大小估算](#附录-b文件大小估算)
14. [附录 C：发布检查清单](#附录-c发布检查清单)

---

## 发布是什么 / 不是什么

| | ✅ 是 | ❌ 不是 |
|---|---|---|
| 目标 | 让陌生用户能 `from_pretrained("Perlous/ClearMind-Base")` 直接用 | 备份训练 checkpoint |
| 产物 | `model.safetensors` + `config.json` + `tokenizer.*` + `README.md` | `final.pth` / `_resume.pth` |
| 格式 | HuggingFace `Qwen3ForCausalLM` 兼容 | ClearMind from-scratch 原生格式 |
| 体积 | fp16 压缩，去除 optimizer state | fp32 训练态，含 optimizer/scheduler |
| 上架平台 | HuggingFace（国际）+ ModelScope（国内） | 网盘 / OSS（这些是分发，不是发布） |

发布的本质是**让模型变成生态里的一等公民** —— vLLM 直接 serve、Ollama 转 GGUF、transformers 自动识别架构、各种 LLM 应用框架（llama.cpp、LM Studio）都能加载。

---

## 核心概念：训练态 vs 发布态

| 维度 | `outputs/dpo/final.pth`（训练态） | `release/<name>/model.safetensors`（发布态） |
|---|---|---|
| 容器 | PyTorch pickle（`.pth`） | safetensors（mmap，零拷贝） |
| 安全 | ❌ 反序列化可执行任意代码（曾被用于供应链攻击） | ✅ 纯数据，无代码执行风险 |
| 精度 | fp32（4 bytes/param） | fp16（2 bytes/param） |
| 键名 | ClearMind 风格（`w_q` / `w_k` / `w_v` / `w_o`） | Qwen3 标准（`q_proj` / `k_proj` / `v_proj` / `o_proj`） |
| `transformers` 自动识别 | ❌（`from_pretrained` 不认 `.pth`） | ✅ |
| 配套 | 仅权重 dict（可能含 optimizer state） | + `config.json` + `tokenizer.*` + `generation_config.json` + `README.md` |
| 体积（small 13.23M params） | ~60 MB | **31.4 MB** |

**`scripts/convert_to_qwen3.py` 做的就是这两态之间的转换**：键名映射 + dtype 量化 + 元数据写出 + tokenizer 拷贝。权重一字未丢，只是换了容器和命名约定。

> **常见误解**：以为不上传 `.pth` 是 bug。实际上 HF/ModelScope 标准就是 safetensors，`.pth` 留在归档里做训练 reproducibility 备份就好（参见 [Phase 1 § 备份策略](#13-备份策略)）。

---

## Phase 1 — 准备工作

### 1.1 环境对齐（CLAUDE.md 强制）

CLAUDE.md 第一原则：

> `transformers>=4.40,<5`：5.x 在 chat_template / from_pretrained 序列化等处不向后兼容；本地与 AutoDL 必须对齐到 4.x。

| 组件 | 推荐版本 | 备注 |
|---|---|---|
| Python | **3.12.x** | 3.13/3.14 上 ML 生态轮子不齐；3.10/3.11 也可 |
| torch | `>=2.1,<3` | 2.5+ 与 transformers 4.5x 配合最稳 |
| transformers | **`>=4.40,<5`** | **不可用 5.x** |
| huggingface_hub | `>=0.24,<2` | 1.x API 已稳定 |
| safetensors | `>=0.4` | |
| modelscope | `>=1.13` | 双发必装 |

**新机器从零搭（推荐 uv，速度比 pip 快 10×）**：

```bash
cd /path/to/clear-mind

# 备份现有 venv（如有）
[ -d venv ] && mv venv venv.bak

# 用 uv 建 Python 3.12 venv
uv venv --python python3.12 venv

# 装 requirements.txt
uv pip install --python ./venv/bin/python -r requirements.txt

# 自检：transformers 必须是 4.x
./venv/bin/python -c "
import sys, torch, transformers, safetensors, huggingface_hub, modelscope
print('Python:', sys.version.split()[0])
for m in ['torch','transformers','safetensors','huggingface_hub','modelscope']:
    v = __import__(m).__version__
    ok = '✅' if (m != 'transformers' or int(v.split('.')[0]) < 5) else '❌'
    print(f'  {ok} {m:20s} {v}')
"
```

**预期输出**（关键看 transformers）：
```
Python: 3.12.13
  ✅ torch                2.x.y
  ✅ transformers         4.5x.y      ← 必须 < 5
  ✅ safetensors          0.x.y
  ✅ huggingface_hub      0.3x or 1.x
  ✅ modelscope           1.3x.y
```

### 1.2 创建远程仓库

**HuggingFace**（国际）：
1. 打开 https://huggingface.co/new
2. 仓库名建议：`<username>/ClearMind-<Tier>`（Tier ∈ {Small, Base, Plus}）
3. 选择 `Public`（除非要先私有冒烟）
4. License 选 `apache-2.0`（与 `release.model_name` 在 yaml 里的设置一致）

**ModelScope**（国内）：
1. 打开 https://www.modelscope.cn/models/create
2. 仓库名同 HF；中文名可填 `ClearMind-Base 中文小模型`
3. License `Apache License 2.0`
4. 模型类型选 `Text Generation` / NLP

> **用户名可能不同**：HF 与 ModelScope 是两个账号体系，如本项目作者 HF 是 `Perlous`、ModelScope 是 `Perlou`。**先别假设一致**，注册各自查一下。

### 1.3 备份策略

`.pth` 不上传 Hub，但要保管好。三层备份：

| 位置 | 用途 | 是否 Git 追踪 |
|---|---|---|
| AutoDL `~/autodl-tmp/clear-mind/outputs/{pretrain,sft,dpo}/final.pth` | 训练机本地，被 `save_outputs.sh` 归档 | ❌ |
| 本地 `release/clearmind-<tier>-YYYYMMDD-HHMM/` | 长期归档，含 sha256 manifest + eval/log | ❌（gitignore） |
| 云盘 / OSS（建议） | 长期保存 + 分享研究复现 | ❌ |

**绝不**把 `.pth` 推到 Git LFS — 体积太大、每次训练都会变、对合作者无价值。

### 1.4 凭证

**HF Token**：https://huggingface.co/settings/tokens → New token → `write` 权限。

```bash
# 持久登录（推荐）
./venv/bin/hf auth login          # 粘贴 token，Add to git credential? Y
./venv/bin/hf auth whoami         # 确认

# 或环境变量（不持久）
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

**ModelScope Token**：https://www.modelscope.cn/my/myaccesstoken → 创建。

```bash
# ModelScope 没有持久 CLI 登录，每次脚本都要传 token
export MODELSCOPE_API_TOKEN=ms_xxxxxxxxxxxxxxxx
```

> ⚠️ **AutoDL 上需要学术加速**才能访问 huggingface.co：
> ```bash
> source /etc/network_turbo            # 仅当前 shell
> hf auth login
> unset http_proxy https_proxy         # 跑训练前关掉，避免污染
> ```
> ModelScope 在国内直连，**不需要加速**。

---

## Phase 2 — 转 Qwen3 兼容格式

### 2.1 拿到归档

AutoDL 训完会有 `release/clearmind-<tier>-YYYYMMDD-HHMM/` 归档（由 `scripts/autodl/save_outputs.sh` 产出），结构：

```
release/clearmind-base-20260612-1430/
├── checkpoints/
│   ├── pretrain/{final.pth, _resume.pth}
│   ├── sft/{final.pth, _resume.pth}
│   └── dpo/{final.pth, _resume.pth}    ← 我们要转的就是这个
├── configs/main.yaml                    ← 训练用的实际配置
├── eval/{generation_samples.txt, perplexity.txt}
├── logs/clearmind-base-all.log
├── tokenizer/minimind/{tokenizer.json, tokenizer_config.json}
└── manifest.txt                          ← 含 sha256
```

把这个目录从 AutoDL 拉到本地 `release/` 下（`scp` / `rsync` 都行）。

### 2.2 转换命令

**核心命令**：
```bash
./venv/bin/python scripts/convert_to_qwen3.py \
    --input  release/clearmind-base-YYYYMMDD-HHMM/checkpoints/dpo/final.pth \
    --config release/clearmind-base-YYYYMMDD-HHMM/configs/main.yaml \
    --output release/clearmind-base \
    --dtype  fp16 \
    --model_name ClearMind-Base
```

**参数说明**：

| 参数 | 含义 | 推荐值 |
|---|---|---|
| `--input` / `-i` | 训练态权重（`final.pth` 或 `_resume.pth`） | 用 dpo 阶段的 final.pth（最终对齐版本） |
| `--config` / `-c` | 训练用的 yaml（决定架构） | **用归档里的 yaml**，不要用仓库根目录的 yaml（防止配置漂移） |
| `--output` / `-o` | 发布目录 | `release/clearmind-<tier>` |
| `--dtype` | 输出权重精度 | `fp16`（推荐）/ `bf16` / `fp32` |
| `--model_name` | HF config 里的 `_name_or_path` | `ClearMind-Base` |

### 2.3 产物清单

转换后 `release/clearmind-base/` 应当至少包含：

| 文件 | 大小（base 68.8M） | 来源 | 用途 |
|---|---|---|---|
| `model.safetensors` | ~138 MB (fp16) | dpo/final.pth 转换 | 权重 |
| `config.json` | <1 KB | yaml → Qwen3 schema | HF 架构识别 |
| `generation_config.json` | <1 KB | yaml.generation | 默认采样参数 |
| `configuration.json` | <1 KB | ModelScope 需要 | MS 架构识别 |
| `tokenizer.json` | ~440 KB | 复制自 `tokenizer/minimind/` | tokenizer |
| `tokenizer_config.json` | ~13 KB | 含 chat_template | ChatML 格式渲染 |
| `README.md` | ~200 B | 自动生成（占位） | 模型卡（**建议手工补完整**，见 § 模型卡模板） |

> **注意 `configuration.json` vs `config.json`**：
> - `config.json` 是 HF 标准
> - `configuration.json` 是 ModelScope 标准（内容近似但 schema 不同）
>
> `convert_to_qwen3.py` 会同时生成两份，一份目录可以同时上 HF 和 MS。

---

## Phase 3 — 本地加载验证

转换完**必须本地验证**再 push，否则上去发现挂了又得删 commit。

### 3.1 基础加载（CPU 即可，~10 秒）

```bash
./venv/bin/python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

REPO = "release/clearmind-base"

tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(
    REPO, trust_remote_code=True, dtype=torch.float32   # transformers 4.5x 用 dtype，不是 torch_dtype
)
print(f"✅ Loaded {sum(p.numel() for p in m.parameters())/1e6:.2f}M params")

# 关键：return_token_type_ids=False（Qwen3 系不接受 token_type_ids）
inputs = tok("你好", return_tensors="pt", return_token_type_ids=False)
out = m.generate(**inputs, max_new_tokens=30, do_sample=False)
print("📝 Greedy:", tok.decode(out[0], skip_special_tokens=True))
PY
```

**通过标准**：
- 无 traceback
- 输出参数量与预期匹配（参考 [附录 B](#附录-b文件大小估算)）
- 中文输出大致连贯（不是乱码）

### 3.2 ChatML 路径验证（确认 SFT/DPO 对齐生效）

```bash
./venv/bin/python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

REPO = "release/clearmind-base"
tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(REPO, trust_remote_code=True, dtype=torch.float32)

messages = [{"role": "user", "content": "你好，你是谁？"}]
chat_inputs = tok.apply_chat_template(
    messages, return_tensors="pt", add_generation_prompt=True
)
print(f"Prompt token len: {chat_inputs.shape[1]}")
out = m.generate(chat_inputs, max_new_tokens=80, do_sample=False)
gen = tok.decode(out[0][chat_inputs.shape[1]:], skip_special_tokens=True)
print("💬", gen)
PY
```

**通过标准**：
- 能正确渲染 `<|im_start|>` / `<|im_end|>` 等控制 token（不会因为 chat_template 缺失退化成裸 prompt）
- 输出是对话式回答，不是续写文本
- 看一下身份：是否自称 ClearMind 还是 MiniMind？（如果还是 MiniMind，回头修 SFT 数据再发布，详见 [FAQ § 身份污染](#身份污染我是-minimind)）

---

## Phase 4 — 推送到 Hub

### 4.1 HuggingFace（`scripts/push_to_hub.py`）

```bash
./venv/bin/python scripts/push_to_hub.py \
    --model_dir release/clearmind-base \
    --repo Perlous/ClearMind-Base \
    --commit "Initial release: ClearMind-Base 68.8M dense (pretrain+sft+dpo)"
```

**完整参数**：

| 参数 | 必填 | 说明 |
|---|---|---|
| `--model_dir` / `-m` | ✅ | release 目录路径 |
| `--repo` / `-r` | ✅ | `<user>/<repo_name>` |
| `--token` |  | 默认读 `HF_TOKEN` 环境变量或登录态 |
| `--private` |  | 创建私有仓库（默认公开） |
| `--commit` / `-c` |  | commit message |
| `--allow_patterns` |  | 只上传匹配 glob 的文件 |
| `--ignore_patterns` |  | 不上传匹配 glob 的文件 |
| `--dry_run` |  | 检查不上传 |

**首次冒烟建议先 `--private`** 验证，通了再创建公开 repo。

**预期上传时间**（家用 100Mbps 上行）：

| Tier | 体积 (fp16) | 上传时长 |
|---|---|---|
| Small (13.23M) | 31 MB | ~30 秒 |
| Base (68.8M) | 138 MB | 1-2 分钟 |
| Plus (486.3M) | 970 MB | 10-15 分钟 |

### 4.2 ModelScope（`scripts/push_to_modelscope.py`）

```bash
export MODELSCOPE_API_TOKEN=ms_xxxxxxxxxxxxxxxx

./venv/bin/python scripts/push_to_modelscope.py \
    --model_dir release/clearmind-base \
    --repo Perlou/ClearMind-Base \
    --visibility public \
    --license apache-2.0 \
    --commit "Initial release: ClearMind-Base 68.8M dense (pretrain+sft+dpo)"
```

**完整参数**：

| 参数 | 必填 | 说明 |
|---|---|---|
| `--model_dir` / `-m` | ✅ | release 目录 |
| `--repo` / `-r` | ✅ | `<user>/<repo_name>` |
| `--token` |  | 默认读 `MODELSCOPE_API_TOKEN` |
| `--visibility` |  | `public` / `private` |
| `--license` |  | `apache-2.0` / `mit` 等 |
| `--commit` / `-c` |  | commit message |
| `--dry_run` |  | 检查不上传 |

> ModelScope 没有 CLI 持久登录，每次都要环境变量或 `--token`。

### 4.3 README 元数据警告

push HF 时可能看到：
```
UserWarning: Warnings while validating metadata in README.md:
  - empty or missing yaml metadata in repo card
```

**不阻塞上传**，但模型卡上不会显示 license / language / pipeline_tag 等标签。**正式发布前补全 README YAML frontmatter**（见 [§ README.md 模型卡模板](#readmemd-模型卡模板)）。

---

## Phase 5 — 远程圆环验证

push 成功 ≠ 发布成功。**圆环验证**才是真正打通：换一个新目录、用 repo id 拉模型、跑推理，模拟陌生用户视角。

### 5.1 HF 圆环

```bash
mkdir -p /tmp/clearmind-roundtrip-hf && cd /tmp/clearmind-roundtrip-hf

/path/to/clear-mind/venv/bin/python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

REPO = "Perlous/ClearMind-Base"
print(f"⬇️  从 HF 拉取 {REPO} ...")

tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(REPO, trust_remote_code=True, dtype=torch.float32)
print(f"✅ {sum(p.numel() for p in m.parameters())/1e6:.2f}M params loaded")

inputs = tok("你好", return_tensors="pt", return_token_type_ids=False)
out = m.generate(**inputs, max_new_tokens=30, do_sample=False)
print(f"📝 {tok.decode(out[0], skip_special_tokens=True)[:100]}")
PY
```

### 5.2 ModelScope 圆环

```bash
mkdir -p /tmp/clearmind-roundtrip-ms && cd /tmp/clearmind-roundtrip-ms

/path/to/clear-mind/venv/bin/python - <<'PY'
from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch

REPO = "Perlou/ClearMind-Base"
print(f"⬇️  从 ModelScope 拉取 {REPO} ...")

tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(REPO, trust_remote_code=True, torch_dtype=torch.float32)
print(f"✅ {sum(p.numel() for p in m.parameters())/1e6:.2f}M params loaded")

inputs = tok("你好", return_tensors="pt", return_token_type_ids=False)
out = m.generate(**inputs, max_new_tokens=30, do_sample=False)
print(f"📝 {tok.decode(out[0], skip_special_tokens=True)[:100]}")
PY
```

> ModelScope 的 SDK 可能仍接受老的 `torch_dtype` 参数；如果它告警让你改 `dtype`，跟随。

### 5.3 圆环通过标准

- ✅ 模型从远端拉下来（看到下载进度）
- ✅ 加载无 traceback
- ✅ 参数量与本地一致
- ✅ 中文输出连贯

**任何一步失败，先在本地复现，找到根因再修，不要直接覆盖 push。**

---

## 三档完整命令对照（small / base / plus）

| 阶段 | small (13.23M) | base (68.8M) | plus (486.3M) |
|---|---|---|---|
| 训练配置 | `configs/small.yaml` | `configs/main.yaml` | `configs/plus.yaml` |
| convert `--config` | `configs/small.yaml` | `configs/main.yaml` | `configs/plus.yaml` |
| convert `--output` | `release/clearmind-small` | `release/clearmind-base` | `release/clearmind-plus` |
| convert `--model_name` | `ClearMind-Small` | `ClearMind-Base` | `ClearMind-Plus` |
| HF repo | `<user>/ClearMind-Small` | `<user>/ClearMind-Base` | `<user>/ClearMind-Plus` |
| MS repo | `<user>/ClearMind-Small` | `<user>/ClearMind-Base` | `<user>/ClearMind-Plus` |
| 体积 (fp16) | 31 MB | 138 MB | 970 MB |
| max_seq_len | 512 | 1024 | 1024 |

---

## 一键流水线：`scripts/release.sh`

整条 Phase 2 → Phase 4 已封装成一个脚本：

```bash
# 仅本地转换 + transformers 加载验证（推荐第一次跑）
bash scripts/release.sh base

# 转换 + 验证 + push HF
bash scripts/release.sh base --push-hf Perlous/ClearMind-Base

# 转换 + 验证 + push HF + push MS
bash scripts/release.sh base \
    --push-hf Perlous/ClearMind-Base \
    --push-ms Perlou/ClearMind-Base

# dry-run（看每步会做什么但不真跑）
bash scripts/release.sh base --dry-run

# 改阶段：默认 dpo，可切到 sft / pretrain
bash scripts/release.sh base --stage sft --push-hf Perlous/ClearMind-Base-SFT

# 改精度：默认 fp16，可改 bf16 / fp32
bash scripts/release.sh plus --dtype bf16 --push-hf Perlous/ClearMind-Plus
```

### release.sh 默认行为

1. 从 `outputs/<stage>/final.pth` 取 ckpt（**默认 dpo**）
2. `convert_to_qwen3.py` → `release/clearmind-<scale>/`
3. transformers 加载 + 推理 8 token 验证
4. tar.gz 打包 + sha256
5. 可选 push HF / ModelScope

> ⚠️ release.sh 是从 `outputs/<stage>/final.pth` 读 ckpt 的，**不是从归档目录**。如果你已经 `rm -rf outputs/`，那就要么把归档里的 `final.pth` 复制回 `outputs/`，要么直接调 `convert_to_qwen3.py` + `push_to_hub.py` 不走 release.sh。

---

## README.md 模型卡模板

`convert_to_qwen3.py` 自动生成的 README 是占位文本，**正式发布前替换成下面这个**。HF 和 ModelScope 都认相同的 YAML frontmatter。

```markdown
---
license: apache-2.0
language:
  - zh
  - en
pipeline_tag: text-generation
library_name: transformers
tags:
  - clearmind
  - chinese-llm
  - from-scratch
  - qwen3-compatible
  - minimind-aligned
base_model: jingyaogong/MiniMind3
datasets:
  - jingyaogong/minimind_dataset
model-index:
  - name: ClearMind-Base
    results:
      - task: { type: text-generation }
        dataset: { type: pretrain_t2t_mini, name: minimind pretrain mini }
        metrics:
          - { type: perplexity, value: 7.82, name: PPL (pretrain stage) }
---

# ClearMind-Base (68.8M)

ClearMind-Base 是一个从零训练的中文 dense 小语言模型，参数量 68.8M，遵循 Qwen3 架构契约，可直接通过 `transformers` / `vLLM` / `Ollama (GGUF)` 使用。

## 与 MiniMind 的关系

ClearMind 复用了 [MiniMind](https://github.com/jingyaogong/minimind) 的 tokenizer（vocab=6400，含 `<|im_start|>` / `<|im_end|>` / `<tool_call>` / `<think>` 与 16 个 buffer token）与数据生态，但在以下方面做了升级：

- **架构**：QK-Norm + RoPE θ=1e6 + YaRN 外推
- **训练工程**：BaseTrainer 抽象、val split + EarlyStopping、参数分组 weight decay、AMP 新 API、SWA、torch.compile 兼容 LoRA、fused AdamW、DDP no_sync、activation checkpointing
- **修复**：attention_mask `0*inf=NaN`、SFT loss-mask BPE 边界错位、RoPE buffer 共享、原子 checkpoint、RMSNorm bf16 dtype 一致性

## 用法

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

repo = "Perlous/ClearMind-Base"
tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, dtype=torch.float16)

messages = [{"role": "user", "content": "你好，你是谁？"}]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors="pt", return_token_type_ids=False)
out = m.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## 训练数据

| 阶段 | 数据 | 样本数 |
|---|---|---|
| Pretrain | `pretrain_t2t_mini.jsonl` | 1.27M |
| SFT | `sft_t2t_mini.jsonl` | ... |
| DPO | `dpo.jsonl` | ... |

## 评测

| 阶段 | PPL（pretrain_t2t_mini 上） |
|---|---|
| Pretrain | 7.82 |
| SFT | 10.49（分布漂移到 ChatML，正常） |
| DPO | 10.54（不优化 NLL，预期持平） |

## 局限

- 仅 68.8M 参数，复杂推理与长文本生成能力有限
- max_seq_len = 1024
- 训练数据体量受限，知识广度不及百 M 级别开源模型

## License

Apache 2.0
```

---

## 故障排查 FAQ

### `transformers` 5.x 报错 / 行为异常

**症状**：本地转出来的 `release/`，在 colab/kaggle 上加载报 `KeyError: 'rope_scaling'` / chat_template 不渲染 / 莫名 `from_pretrained` 失败。

**根因**：你本地是 transformers 5.x，绝大多数 Hub 用户在 4.x。5.x 的 config 序列化字段与 4.x 不兼容。

**修复**：
```bash
./venv/bin/pip install -U "transformers>=4.40,<5"
# 重新跑 convert_to_qwen3.py 重新生成 release/
```

> 这就是为什么 [Phase 1 § 1.1](#11-环境对齐claudemd-强制) 强制要求 4.x。

---

### `ValueError: model_kwargs are not used by the model: ['token_type_ids']`

**根因**：tokenizer 默认返回 `token_type_ids`（用于 BERT 双输入），但 Qwen3 不接受。

**修复**：调用 tokenizer 时显式禁用：
```python
inputs = tok("你好", return_tensors="pt", return_token_type_ids=False)
```

---

### `torch_dtype` is deprecated! Use `dtype` instead!

**根因**：transformers 4.55+ 把 `from_pretrained(torch_dtype=...)` 改名为 `dtype=...`。

**修复**：把所有 `torch_dtype=torch.float32` 改成 `dtype=torch.float32`。如果是 modelscope SDK 的 `from_pretrained`，老参数名仍可能兼容，跟随它当前的告警调整。

---

### 身份污染："我是 MiniMind"

**症状**：模型回答 "你好，你是谁？" 时自称 MiniMind 而不是 ClearMind。

**根因**：SFT 数据来自 `minimind_dataset`，里面所有自我介绍样本都是 MiniMind 写的，模型 memorize 了身份。

**修复方案**（由轻到重）：

1. **System prompt 兜底（最快）**：发布版的 chat 调用强制带 `system: "你是 ClearMind..."`，但用户空 system 调用时仍会泄露。

2. **追加身份 SFT（推荐）**：构造 200-500 条 ClearMind 自我介绍样本，最后追加一轮 SFT（小 lr、1 epoch）。

3. **数据全局替换 + 重训**：把 SFT 数据里 "MiniMind / minimind / 小明" 全局替换成 "ClearMind"，重新跑 SFT。

> ⚠️ **公开发布前必修**，否则用户体验会很尴尬。

---

### `max_new_tokens (X) ≥ max_seq_len (X)`，临时降到 X/2

**根因**：`src/inference/chat.py:179` 的防御性夹紧。当 `max_new_tokens >= max_seq_len`，budget = max_seq_len - max_new_tokens 会变 0/负，导致 `prompt_ids[-keep:]` 在 keep 为负时返回 `[]` → MPS 上 `torch.full((0,0))` 直接断言失败。

**修复**：在 yaml 里让 `generation.max_new_tokens` 严格 **< `model.max_seq_len`**：

| 配置 | model.max_seq_len | 推荐 generation.max_new_tokens |
|---|---|---|
| `small.yaml` | 512 | 384 |
| `main.yaml` | 1024 | 512 或 768 |
| `plus.yaml` | 1024 | 512 或 768 |

> 已训练好的模型 `max_seq_len` 是固化的（RoPE buffer 大小固定），改 yaml 不能让现有 ckpt 长出 1024 的能力，只能控制推理时的 `max_new_tokens`。

---

### AutoDL 上 `hf auth login` hang

**症状**：粘 token 后卡死，按 Ctrl+C 才报 `KeyboardInterrupt`，traceback 在 `whoami(token)` 的 `urllib3.create_connection`。

**根因**：AutoDL 在国内，无法直连 huggingface.co。

**修复**：
```bash
source /etc/network_turbo            # 学术加速
hf auth login
unset http_proxy https_proxy         # 跑训练前关掉
```

或用镜像（仅适合下载，不支持私有 push）：
```bash
export HF_ENDPOINT=https://hf-mirror.com
hf auth login
```

---

### push 时 README 元数据警告

**症状**：
```
UserWarning: empty or missing yaml metadata in repo card
```

**根因**：`convert_to_qwen3.py` 生成的 README.md 是占位文本，没有 HF 标准 YAML frontmatter。

**修复**：用 [§ README.md 模型卡模板](#readmemd-模型卡模板) 替换。**不阻塞上传**，但模型卡显示效果差。

---

### `model.safetensors` 比 `final.pth` 小一半

不是 bug。原因：

- `final.pth` 是 fp32（4 bytes/param）+ 可能含 optimizer state
- `model.safetensors` 是 fp16（2 bytes/param）+ 纯权重

体积约 1/2 是预期，参见 [附录 B](#附录-b文件大小估算) 的精确公式。

---

### 加载后参数量 < manifest.txt 报告

**症状**：manifest 写 15.68M，加载后 `sum(p.numel())` 显示 13.23M，差 2.45M。

**根因**：tied embeddings。input_embedding 和 lm_head 共用同一份权重，文件里分两份存（HF 标准布局），运行时被识别为同一份。

差值正好等于 vocab_size × d_model：
- small: 6400 × 384 = 2,457,600 ≈ 2.46M ✅

不是 bug。

---

### ModelScope push 报 `repo already exists`

**根因**：脚本试图 `create_model` 但仓库已存在且 schema 不匹配。

**修复**：
- 网页删 repo 重建
- 或换名字
- 或编辑 `push_to_modelscope.py`，让它在 `get_model` 成功时跳过 `create_model`

---

### 上传中途断网

**HF**：`huggingface_hub.upload_folder` 默认会 resume，重新跑命令即可。
**ModelScope**：可能需要手工 `git push` 或重新 `push_model`。

大文件建议网络稳定时再 push，或在 AutoDL（push HF 走学术加速）/ 国内云主机（push MS 直连）上做。

---

## 附录 A：版本对齐矩阵

发布管线参与者必须严格对齐到下表。**任何环节用 5.x 都会让陌生用户加载失败。**

| 角色 | Python | torch | transformers | 备注 |
|---|---|---|---|---|
| AutoDL 训练机 | 3.10 / 3.12 | `>=2.1` | **`>=4.40,<5`** | preflight.sh 第 8 步会跑 pytest，缺 transformers 直接红灯 |
| 本地发布机 | **3.12** | `>=2.1` | **`>=4.40,<5`** | 与 AutoDL 对齐，避免序列化漂移 |
| Hub 用户（colab/kaggle） | 3.10+ | `>=2.1` | **`>=4.40,<5`** | 我们没法控制，但 4.x 是社区默认 |
| huggingface_hub | - | - | - | `>=0.24,<2` |
| safetensors | - | - | - | `>=0.4` |
| modelscope | - | - | - | `>=1.13` |

---

## 附录 B：文件大小估算

**公式**：`体积 ≈ 文件参数总数 × bytes_per_param + 元数据开销 (~1 MB)`

| dtype | bytes/param | 经验系数 |
|---|---|---|
| fp32 | 4 | × 4 |
| **fp16** ⭐ | **2** | × 2 |
| bf16 | 2 | × 2 |
| int8（量化） | 1 | × 1 |
| int4 (q4_0) | 0.5 | × 0.5 |

**注意：文件参数 ≠ 加载后参数**（tied embeddings 差异）。

`文件参数 = 加载后参数 + vocab_size × d_model`（embedding/lm_head 各存一份）

**实际尺寸表**：

| Tier | 加载后参数 | 文件参数 | fp16 体积 | int4 量化体积 (q4_0) |
|---|---|---|---|---|
| Tiny (~0.5M) | 0.5M | ~1.3M | ~3 MB | ~1 MB |
| **Small** | 13.23M | 15.68M | **31 MB** | ~9 MB |
| **Base** | ~64M | 68.8M | **138 MB** | ~37 MB |
| **Plus** | ~480M | 486.3M | **973 MB** | ~250 MB |

---

## 附录 C：发布检查清单

push 公开 repo 前**逐项打勾**：

### 配置
- [ ] `configs/<tier>.yaml` 的 `max_seq_len` 与 CLAUDE.md 规格表一致（main/plus = 1024）
- [ ] `configs/<tier>.yaml` 的 `generation.max_new_tokens` < `model.max_seq_len`
- [ ] 训练用的 yaml 与归档 `release/.../configs/` 里的 yaml 一致

### 训练产物
- [ ] `outputs/dpo/final.pth` 存在且非零字节
- [ ] `eval/perplexity.txt` PPL 数字合理（pretrain < 20，sft/dpo 略高于 pretrain 是正常的）
- [ ] `eval/generation_samples.txt` 输出可读（不是乱码）
- [ ] manifest.txt 含 sha256，可校验完整性

### 转换
- [ ] `release/clearmind-<tier>/` 存在
- [ ] 含 `model.safetensors` + `config.json` + `generation_config.json` + `configuration.json` + `tokenizer.json` + `tokenizer_config.json`
- [ ] `config.json` 的 `architectures` = `["Qwen3ForCausalLM"]`
- [ ] 文件总大小符合 [附录 B](#附录-b文件大小估算)

### 本地验证
- [ ] `transformers.AutoModelForCausalLM.from_pretrained` 加载无 traceback
- [ ] 参数量与预期匹配（差值 ≤ vocab_size × d_model）
- [ ] Greedy 采样输出连贯中文
- [ ] `apply_chat_template` 路径输出对话格式（不是续写）
- [ ] **身份不自称 MiniMind**（如果 small 还在污染，至少 main/plus 必须修）

### 模型卡
- [ ] `release/clearmind-<tier>/README.md` 含 YAML frontmatter（license / language / pipeline_tag / tags）
- [ ] 含用法 code block
- [ ] 含训练数据来源、评测数字、局限性
- [ ] 含 base_model: `jingyaogong/MiniMind3` 致谢

### Push
- [ ] HF repo 已建（`<user>/ClearMind-<Tier>`）
- [ ] MS repo 已建
- [ ] HF / MS token 在环境变量或登录态
- [ ] `--commit` 信息描述清楚版本（"Initial release: ..." / "Bump tokenizer fix: ..."）
- [ ] 首次冒烟用 `--private` 验证

### 圆环
- [ ] `/tmp/` 下用 repo id 拉模型成功
- [ ] HF 端拉取 + 推理通过
- [ ] ModelScope 端拉取 + 推理通过
- [ ] 模型卡页面元数据标签正常显示

### 公开
- [ ] HF repo 改 public
- [ ] MS repo 改 public
- [ ] 在项目 README / 个人主页贴上链接
- [ ] 在 CLAUDE.md / PROGRESS_TRACKER.md 记录发布日期与 commit hash
