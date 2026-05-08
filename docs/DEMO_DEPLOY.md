# 🎬 ClearMind Demo 部署指南（ModelScope 创空间）

> 把 ClearMind 模型在线对话演示部署到 **ModelScope 创空间（Studio）**，让任何人通过浏览器访问 demo。
>
> 本文件聚焦 **demo 网站发布**（演示通道）。如果你要发布的是模型权重本身（让其他人 `from_pretrained` 加载），看 [`RELEASE_GUIDE.md`](RELEASE_GUIDE.md)。如果你要的是推理服务化（OpenAI 兼容 API、Docker），看 [`DEPLOY.md`](DEPLOY.md)。

---

## 📋 目录

1. [概览：demo 是什么](#概览demo-是什么)
2. [设计决策](#设计决策)
3. [`space/` 目录结构](#space-目录结构)
4. [Phase 1 — 本地开发与冒烟](#phase-1--本地开发与冒烟)
5. [Phase 2 — 创建 ModelScope 创空间](#phase-2--创建-modelscope-创空间)
6. [Phase 3 — 推送代码与构建](#phase-3--推送代码与构建)
7. [Phase 4 — 验证与发布](#phase-4--验证与发布)
8. [上架后的运维](#上架后的运维)
9. [base/plus 训完后的迁移](#baseplus-训完后的迁移)
10. [故障排查 FAQ](#故障排查-faq)
11. [附录 A — 成本与性能](#附录-a--成本与性能)
12. [附录 B — 文件清单速查](#附录-b--文件清单速查)
13. [附录 C — 后续扩展](#附录-c--后续扩展)

---

## 概览：demo 是什么

| | ✅ 是 | ❌ 不是 |
|---|---|---|
| 形态 | 浏览器可访问的对话页面（Streamlit） | API 服务 / SDK / Python 包 |
| 受众 | 想"先看看效果"的潜在用户 / 评审 / 投资人 / 媒体 | 工程师集成 / 二次开发 |
| 后端 | 远程拉 ModelScope 上的 ClearMind 权重 | 本地 .pth 文件 |
| 平台 | **ModelScope 创空间**（gongjy/MiniMind 同款） | HF Spaces（本期不做，预留双发能力） |
| 月费 | **¥0**（免费 CPU 档） | - |

参照对象：[gongjy 的 MiniMind 创空间](https://www.modelscope.cn/studios/gongjy/MiniMind) — 我们要做一个一样形态的产物。

---

## 设计决策

| 决策 | 选择 | 理由 |
|---|---|---|
| Web 框架 | **Streamlit** | 跟 minimind 对齐；ModelScope 创空间一等公民支持；写起来比 Gradio 紧凑 |
| 部署平台 | **ModelScope 创空间** | 国内直连快；CPU 永久免费；与权重仓库同账户，迁移成本低 |
| 模型加载 | `modelscope.AutoModelForCausalLM.from_pretrained(repo_id)` | 直接读 ModelScope 上的 model.safetensors，无需把权重打进 demo 仓库 |
| 资源档 | CPU 2 vCPU + 8 GB | base/plus 都跑得动；月费 ¥0 |
| sidebar 选项 | 仅 **Plus + Base**，small 不展示 | small 是冒烟规格，不对外宣传 |
| 当前权重 | base/plus 都临时指向 `Perlou/ClearMind-Small` | 走通管线优先；训完只改 2 行 repo id |
| 顶部 padding | `100px`（MS 平台默认） | 避开创空间顶部 banner |

---

## `space/` 目录结构

```
space/
├── app.py                  # Streamlit 主程序（524 行；think + 8 工具 + 双语 + 流式）
├── requirements.txt        # 精简依赖（不引入训练相关）
├── README.md               # MS Studio 元数据 + 模型卡（YAML frontmatter）
├── configuration.json      # ModelScope 平台配置（task / framework / license）
└── .gitignore              # 忽略本地缓存
```

> ⚠️ **`space/` 是被推到 ModelScope 创空间的"独立仓库"**，不要把它跟主项目仓库混在一起 push。主项目 `git push` 走 GitHub，space 走 `modelscope.cn`。

---

## Phase 1 — 本地开发与冒烟

### 1.1 装 streamlit

uv 创建的 venv 默认不带 pip，用 uv 装：

```bash
cd /Users/perlou/Desktop/personal/clear-mind
uv pip install --python ./venv/bin/python streamlit
```

### 1.2 启动

```bash
cd space
CLEARMIND_PLATFORM=ms ../venv/bin/python -m streamlit run app.py \
    --server.port 8501 \
    --server.headless true \
    --browser.gatherUsageStats false
```

打开 http://localhost:8501 。**首次启动会拉 `Perlou/ClearMind-Small` 权重（约 30 秒）**，之后从 `~/.cache/modelscope/hub/` 命中。

### 1.3 冒烟清单

| 检查项 | 通过标准 |
|---|---|
| 页面加载 | sidebar 显示 Plus + Base 两个选项，主区显示 "🧠 我是 ClearMind-Plus..." |
| 顶部留白 | MS 平台默认 100px padding；本地若想测 0px 设 `CLEARMIND_TOP_PAD=0` |
| 流式输出 | 输入"你好"，能看到逐 token 输出 |
| 思考模式 | 勾选 sidebar 的"思考"，输出能折叠 `<think>` |
| 工具调用 | 勾选"时间"+"数学"，输入相关问题，看到 ToolCalling 蓝框 + ToolCalled 绿框 |
| 多轮 | 历史对话轮次 ≥ 2 时，模型能引用上文 |
| 双语 UI | 切换 中文 / English，全部文案变化 |
| Plus ↔ Base 切换 | sidebar 切换不报错（cache 命中，不重下载） |

### 1.4 调试 tip

```bash
# 实时日志
tail -f /tmp/clearmind-streamlit.log

# 改了 app.py 自动 reload
# 启动时加 --server.runOnSave true（改文件就自动重启）

# 杀掉
kill <PID>
```

### 1.5 环境变量速查

| 变量 | 默认 | 用途 |
|---|---|---|
| `CLEARMIND_PLATFORM` | `ms` | `ms` 走 modelscope SDK，`hf` 走 transformers |
| `CLEARMIND_REPO_BASE` | `Perlou/ClearMind-Small`（占位） | 部署时切到真 base 仓库 |
| `CLEARMIND_REPO_PLUS` | `Perlou/ClearMind-Small`（占位） | 部署时切到真 plus 仓库 |
| `CLEARMIND_TOP_PAD` | `100`（ms 平台）/ `0`（其他） | 顶部 padding 像素，可显式覆盖 |
| `CLEARMIND_USE_ZEROGPU` | `0` | HF Spaces 专用，MS 上无效 |

---

## Phase 2 — 创建 ModelScope 创空间

### 2.1 网页建仓库

1. 浏览器打开 → https://www.modelscope.cn/studios/create
2. 填写：

| 字段 | 值 |
|---|---|
| 创空间名称（英文 ID） | `ClearMind-Demo` |
| 中文名称 | `ClearMind 在线演示` |
| 简介 | `从零训练的中文 dense 小模型在线演示` |
| 标签 | `LLM` `中文` `from-scratch` |
| **SDK** | **Streamlit** |
| **资源** | **CPU 免费**（2 vCPU + 8 GB） |
| License | `Apache License 2.0` |
| 可见性 | 公开 |

3. 创建后会得到 git 地址：`https://www.modelscope.cn/studios/Perlou/ClearMind-Demo.git`

### 2.2 ModelScope 个人 token

git push 需要鉴权，预先准备：

1. https://www.modelscope.cn/my/myaccesstoken → 创建/复制 access token
2. 不要把 token 写进 git 远端 URL（会落进 reflog 历史），用 `git credential-helper`：

```bash
git config --global credential.helper osxkeychain   # macOS
# Linux 用 store 或 cache
```

第一次 `git push` 会弹用户名/密码（用户名 = 你的 MS 用户名，密码 = access token），输完之后 keychain 帮你记住。

---

## Phase 3 — 推送代码与构建

### 3.1 克隆 + 复制 + 推送

**关键**：在项目外面克隆（避免和主项目 git 冲突），用 `space/` 内容覆盖。

```bash
# 1. 在临时目录克隆
cd /tmp
git clone https://www.modelscope.cn/studios/Perlou/ClearMind-Demo.git
cd ClearMind-Demo

# 2. 复制 space/ 全部内容（包括隐藏文件）
cp -r /Users/perlou/Desktop/personal/clear-mind/space/. .

# 3. 检查
ls -la
# 应该看到 app.py / requirements.txt / README.md / configuration.json / .gitignore

# 4. 提交
git add .
git commit -m "Initial: ClearMind streamlit demo (Plus/Base sidebar, small placeholder)"

# 5. push（首次会弹密码，输 access token）
git push
```

### 3.2 触发构建

push 完成后 ModelScope 会**自动触发构建**，无需手动操作。

构建步骤（平台执行，约 5-15 分钟）：
1. 拉取你 push 的代码
2. 创建 sandbox 容器
3. `pip install -r requirements.txt`（torch + transformers + streamlit 等）
4. 启动 `streamlit run app.py`
5. 状态从"构建中"变成"运行中"

### 3.3 看构建日志

ModelScope Studio 详情页右上角 → "运行日志" / "构建日志"。常见输出：

```
Successfully installed streamlit-1.32.0 ...
You can now view your Streamlit app in your browser.
URL: http://0.0.0.0:7860
```

平台会把内部 7860 映射到你的公网 URL。

---

## Phase 4 — 验证与发布

### 4.1 公网访问

构建成功后，URL 是：

```
https://www.modelscope.cn/studios/Perlou/ClearMind-Demo
```

把这个 URL 当线上版本，过一遍 [Phase 1.3 冒烟清单](#13-冒烟清单)。

### 4.2 重点关注

| 项目 | 在 MS 创空间上的特殊行为 |
|---|---|
| 首次访问 | 容器冷启动 + 模型下载，约 1-2 分钟。**第一个用户体验差，无解** |
| 闲置休眠 | MS 创空间 **30 分钟无访问会自动休眠**，下次访问要冷启 50-60s |
| 顶部 100px 留白 | 应该看到主标题不被平台 banner 遮 |
| 国内访问速度 | 应该 < 200ms TTFB |

### 4.3 二维码 / 分享

ModelScope Studio 详情页有"分享"按钮，可生成二维码发朋友圈/钉钉群。

---

## 上架后的运维

### 切换模型权重（不改代码）

ModelScope Studio 网页 → 「设置」→ 「环境变量」面板：

```
CLEARMIND_REPO_BASE=Perlou/ClearMind-Base
CLEARMIND_REPO_PLUS=Perlou/ClearMind-Plus
```

保存后 Studio 自动重启。**无需 push 代码**。

### 灰度发布

如果想让 Plus 切到新版本但 Base 保持旧版本，分别设两个变量。

如果想紧急回退：把 `CLEARMIND_REPO_BASE` 设回 `Perlou/ClearMind-Small`。

### 监控

ModelScope Studio 详情页底部有：
- 运行日志（`streamlit` stdout/stderr）
- 资源监控（CPU / 内存使用曲线）
- 访问统计（PV / UV）

如果看到 OOM kill：免费 8GB 内存装不下 Plus（486M fp16 + chat history + streamlit overhead），需要：
- 升级到 16GB 配置档（约 ¥0.1/h）
- 或在 `app.py` 里加载时显式 `low_cpu_mem_usage=True`

### 重新构建

push 新 commit 自动触发；也可在网页「重新构建」按钮手动触发。

---

## base/plus 训完后的迁移

整条流程**只改 1-2 行**，不需要重写任何代码。

### 路线 A：环境变量（推荐，零代码改动，秒级生效）

1. AutoDL 训完 → 跑 `bash scripts/release.sh base --push-ms Perlou/ClearMind-Base`
2. ModelScope Studio → 设置 → 环境变量：
   ```
   CLEARMIND_REPO_BASE=Perlou/ClearMind-Base
   ```
3. 保存 → Studio 自动重启 → 用户下一次切到 Base 就拉真 base 权重

### 路线 B：改代码 push（永久生效，会重新构建）

1. 改 `space/app.py` 的 `DEFAULT_REPOS["ms"]["ClearMind-Base"]` 字符串
2. push 到 MS Studio repo（参照 [Phase 3](#phase-3--推送代码与构建)）
3. 自动重新构建

### 推荐顺序

1. **先用 Plus 训完做路线 A** → 验证 demo 上 Plus 体验是否符合预期
2. 没问题后再做 base 的迁移
3. 都稳定后做路线 B 把 `DEFAULT_REPOS` 写死，移除环境变量

---

## 故障排查 FAQ

### MS Studio 构建一直失败 / 超时

**排查顺序**：

1. **看构建日志**：MS Studio 详情页 → "构建日志"
2. **streamlit 版本**：`requirements.txt` 锁的 `streamlit==1.32.0` 在 MS 上构建稳定。如果失败试 `streamlit==1.28.0`
3. **torch 太新**：MS 平台 base image 可能跟不上 torch 2.11。试 `torch>=2.1.0,<2.6`
4. **modelscope SDK 版本**：MS 自己的环境装 modelscope 是默认；如果跟我们 requirements.txt 锁的版本冲突，删掉这行，让平台用自带版本

---

### Demo 起来了但模型加载失败

**症状**：streamlit 转圈 → 报错 "Model not found"

**排查**：

1. **模型 repo 可见性**：去 https://www.modelscope.cn/models/Perlou/ClearMind-Small 确认是 public，不是 private
2. **trust_remote_code**：app.py 里已经传 `trust_remote_code=True`，不会有这个问题
3. **平台模式错位**：MS Studio 上 `CLEARMIND_PLATFORM` 必须是 `ms`，否则会用 transformers 走 HF（HF 在国内被墙）

---

### 顶部 100px 留白不对

**情况 1：本地看到 100px 空白**：正常，这是模拟 MS 平台。设 `CLEARMIND_TOP_PAD=0` 可以临时调成 0。

**情况 2：MS 上仍被遮**：MS 平台 banner 高度可能更新过。改 `space/app.py` 顶部：
```python
TOP_PAD_PX = int(os.environ.get("CLEARMIND_TOP_PAD", "120" if PLATFORM == "ms" else "0"))
```
push 重新构建，或用环境变量 `CLEARMIND_TOP_PAD=120` 立即生效。

---

### 8GB 内存装不下 Plus

**症状**：日志看到 `OOMKilled` / 容器自动重启

**修复**（按推荐度）：

1. **改加载方式**：app.py 里 `from_pretrained` 加 `low_cpu_mem_usage=True, torch_dtype=torch.float16`
2. **GGUF 量化路径**：把 plus 转 q4_0 GGUF（~250MB），用 llama-cpp-python 加载（需要重写 app.py 的 `load_model_tokenizer`）
3. **升级配置档**：网页面板 → 资源 → 16GB（按时长收费）

---

### "为什么这么聪明？"（错觉警告）

如果你或你的观众觉得 demo 里的模型回答远超 13M 模型预期，**几乎可以肯定是错觉**，原因：

1. `temperature=0.9 + top_p=0.85` 引入流畅性
2. system prompt 把模型定位到"助手"角色
3. SFT 数据里的模板答案被 memorize
4. 思考链装饰增加感知聪明度

**如何识破**：让 demo 跑 stress test：

| 测试输入 | small 真实表现 |
|---|---|
| `鲁迅死于哪年？` | 编造年份或答非所问 |
| `小明有 3 个苹果，吃了 2 个，又买了 5 个` | 算错或绕开 |
| `用 5 句话写关于秋天的诗，每句必须有"叶"字` | 凑不齐 / 漏字 |
| 关闭思考 + temperature 0.3 + `1+1=?` | 莫名其妙的输出 |

---

### push 到 MS Studio 仓库被拒（401 / 403）

**根因**：access token 缺权限或过期。

**修复**：

```bash
# 重新拿 token
# https://www.modelscope.cn/my/myaccesstoken

# 删旧 keychain 凭证
git config --global --unset credential.helper
# 或 macOS: 钥匙串访问 → 搜 modelscope → 删除

# 重新 push（会重新弹密码框）
cd /tmp/ClearMind-Demo
git push
```

---

### 想本地完全复现 MS Studio 环境

```bash
# 1. 装跟 MS 一致的依赖（不要装训练依赖）
cd /tmp/ClearMind-Demo
uv pip install --python /Users/perlou/Desktop/personal/clear-mind/venv/bin/python -r requirements.txt

# 2. 设环境变量
export CLEARMIND_PLATFORM=ms
export CLEARMIND_TOP_PAD=100

# 3. 跑
/Users/perlou/Desktop/personal/clear-mind/venv/bin/python -m streamlit run app.py \
    --server.port 8501 --server.headless true
```

---

## 附录 A — 成本与性能

### 月费对照（demo 使用场景）

| 方案 | 月费 | 用户体验 | 维护成本 |
|---|---|---|---|
| **MS 创空间 CPU 免费** ⭐ | **¥0** | base/small 流畅，plus 慢 60-120s | 极低 |
| MS 创空间 T4 GPU + 闲置自动停 | ~¥30-100 | 全档秒级响应 | 低（设个空闲休眠） |
| MS 创空间 T4 GPU 24/7 | ~¥300 | 全档极致体验 | 低 |
| 自部署阿里云 t6 | ~¥30 | base 勉强，plus 不能 | 中（自己维护） |
| 自部署阿里云 GPU 共享 | ~¥1500 | 极致 | 高 |

**当前选择**：免费 CPU 档，¥0/月。如果 plus 训完发现 60s+ 延迟用户接受不了，升级 T4。

### CPU 推理实测预期

| 模型 | 单条响应（200 token） | 用户体验 |
|---|---|---|
| Small (13M) | 5-10 s | ✅ 流畅 |
| Base (68M) | 10-20 s | ⚠️ 可接受 |
| Plus (486M) | 60-120 s | ❌ 太慢 |

> 数字基于 transformers fp16 推理 + 2vCPU 估算，实际 ±50%。

---

## 附录 B — 文件清单速查

### space/app.py 关键变量

| 变量 | 默认 | 含义 |
|---|---|---|
| `PLATFORM` | `ms` | 平台分支（ms / hf） |
| `DEFAULT_REPOS` | dict | 各规格的默认 repo id |
| `REPOS` | dict | 经过环境变量覆盖的最终 repo id |
| `TOP_PAD_PX` | 100 (ms) / 0 (其他) | 顶部 padding 像素 |
| `USE_ZEROGPU` | `False` | HF Spaces 专用 |
| `TOOLS` | list[8] | 8 个 mock 工具定义 |

### space/requirements.txt 关键依赖

| 包 | 锁定版本 | 用途 |
|---|---|---|
| `streamlit` | `==1.32.0` | UI 框架 |
| `torch` | `>=2.1.0` | 推理 |
| `transformers` | `>=4.40.0,<5` | **5.x 不兼容**（CLAUDE.md 强制） |
| `modelscope` | `>=1.13.0` | MS 平台 SDK |
| `safetensors` | `>=0.4.0` | 权重格式 |

### space/README.md YAML frontmatter 字段

| 字段 | 含义 | MS Studio 是否使用 |
|---|---|---|
| `license` | Apache License 2.0 | ✅ |
| `tags` | 模型标签 | ✅（搜索） |
| `language` | 支持语言 | ✅ |
| `sdk` | streamlit | ✅（决定容器类型） |
| `sdk_version` | 1.32.0 | ✅ |
| `app_file` | app.py | ✅（构建入口） |

---

## 附录 C — 后续扩展

### C.1 双发到 HuggingFace Spaces（国际版）

`app.py` 已经预留了 HF 平台分支。流程：

1. 浏览器建 Space：https://huggingface.co/new-space → SDK Streamlit, Hardware CPU basic
2. 把 `space/` 推到 HF Space repo
3. Space 设置环境变量 `CLEARMIND_PLATFORM=hf`
4. 等构建完成

**Plus 体验优化**：HF 上设 `CLEARMIND_USE_ZEROGPU=1` + README YAML 加 `hardware: zero-a10g`，免费拿 A10G GPU（每用户每天约 5 分钟配额）。

### C.2 自定义工具

`space/app.py` 的 `TOOLS` list 是从 minimind 抄来的 8 个 mock。要换成真工具：

1. 在 `TOOLS` 里加新条目（参考 OpenAI function calling schema）
2. 在 `execute_tool()` 里实现该工具的真实逻辑（API 调用 / DB 查询）
3. 工具失败必须 return `{"error": "..."}`，不要抛异常（chat 会卡死）

### C.3 加 stress test 用例 button

侧边栏加几个"试试这些问题"快速按钮，让用户一键展示模型短板：
- `鲁迅死于哪年？`
- `小明苹果题`
- `5 句叶字诗`

这样观众一进来就能看到模型真实能力，避免被流畅性误导。

### C.4 接入 wandb / swanlab 做 demo 流量监控

Studio 自带的访问统计很基础。如果要细粒度看用户问什么、模型答什么：

1. `space/app.py` 里加日志（每条对话写到 `logs/chat.jsonl`）
2. 用 wandb 的 Tables API 把 jsonl 同步上去
3. dashboard 看热门问题、低分回答、平均生成长度

注意 GDPR/隐私：MS Studio 用户对话默认平台不留存，自加日志要在 README 显式声明并提供删除入口。

---

## 一行命令速查

```bash
# 本地开发
cd space && CLEARMIND_PLATFORM=ms ../venv/bin/python -m streamlit run app.py --server.port 8501

# 推到 MS Studio（首次）
cd /tmp && git clone https://www.modelscope.cn/studios/Perlou/ClearMind-Demo.git && \
  cd ClearMind-Demo && cp -r /Users/perlou/Desktop/personal/clear-mind/space/. . && \
  git add . && git commit -m "Initial demo" && git push

# base/plus 训完迁移（路线 A：网页设环境变量，0 代码改动）
# CLEARMIND_REPO_BASE=Perlou/ClearMind-Base
# CLEARMIND_REPO_PLUS=Perlou/ClearMind-Plus

# 强制 100px 顶部留白
export CLEARMIND_TOP_PAD=100

# 杀本地 streamlit
pkill -f "streamlit run app.py"
```
