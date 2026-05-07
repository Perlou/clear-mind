"""GPT 模型单元测试"""

import torch
import torch.nn.functional as F

from model import GPT


class TestGPT:
    """测试 GPT 模型"""

    def test_forward_shape(self, tiny_model, tiny_config, sample_input_ids):
        """前向传播输出维度应正确"""
        with torch.no_grad():
            logits, loss, _ = tiny_model(sample_input_ids)

        batch, seq_len = sample_input_ids.shape
        assert logits.shape == (batch, seq_len, tiny_config.vocab_size)
        assert loss is None  # 未提供 targets 时 loss 为 None

    def test_forward_with_targets(self, tiny_model, tiny_config, sample_input_ids):
        """提供 targets 时应计算 loss"""
        targets = sample_input_ids.clone()
        with torch.no_grad():
            logits, loss, _ = tiny_model(sample_input_ids, targets)

        assert loss is not None
        assert loss.dim() == 0  # scalar
        assert loss.item() > 0  # loss 应为正数

    def test_forward_does_next_token_shift(self, tiny_config):
        """forward 必须做 next-token shift：CE(logits[:, :-1], targets[:, 1:])

        这是核心回归测试。曾经的 bug 是 forward 直接 CE(logits[t], targets[t])，
        叠加 weight-tying 后模型只需学一个近似 identity 映射就能让 loss → 0、
        PPL → 1.00，但聊天阶段完全失能（空响应）。一旦 forward 不做 shift，
        本测试会立刻失败。

        正向：手动构造的"shifted CE"必须与 forward 内部计算 **完全一致**。
        负向：构造一个"仅第 0 位有效 label"的样本——shift 后该位会落到位置 -1
              被丢弃，所有 shifted_labels 全 -100、loss=0；若 forward 没有 shift，
              则会在位置 0 上计算到非零 CE，loss > 0。
        """
        torch.manual_seed(42)
        model = GPT(tiny_config)
        model.eval()

        # ---------- 正向：与手动 shifted CE 数值一致 ----------
        input_ids = torch.randint(0, tiny_config.vocab_size, (2, 8))
        targets = torch.randint(0, tiny_config.vocab_size, (2, 8))

        with torch.no_grad():
            logits, loss, _ = model(input_ids, targets)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_targets = shift_targets.view(-1)
            loss_sum = F.cross_entropy(
                flat_logits, flat_targets, ignore_index=-100, reduction="sum"
            )
            valid_count = (flat_targets != -100).sum().clamp(min=1)
            expected_shifted = loss_sum / valid_count

        assert torch.allclose(loss, expected_shifted, atol=1e-6), (
            f"forward 必须用 shifted CE；loss={loss.item():.6f} "
            f"expected={expected_shifted.item():.6f}"
        )

        # ---------- 负向：仅第 0 位有效 label 时 shift 必然全 -100 → loss=0 ----------
        # 这是个对 shift 有/无极敏感的结构化判别：
        #   shift 后 labels[:, 1:] 全 -100，valid_count=0→clamp=1，loss=0
        #   若 forward 不 shift，labels[:, 0] 上仍有有效 label，loss > 0（非零）
        only_first_input = torch.randint(0, tiny_config.vocab_size, (1, 8))
        only_first_target = torch.full((1, 8), -100, dtype=torch.long)
        only_first_target[0, 0] = 0  # 一个任意有效 token id
        with torch.no_grad():
            _, only_first_loss, _ = model(only_first_input, only_first_target)
        assert only_first_loss.item() == 0.0, (
            "forward 没有做 shift —— 仅第 0 位有效 label 的样本应被 shift 完全丢弃，"
            f"但得到 loss={only_first_loss.item():.6f}（非零意味着 PPL 会塌成 1.00 类 bug 复发）"
        )

    def test_forward_seq_len_one_does_not_crash(self, tiny_config):
        """seq_len=1 时 shift 后 shape=[B,0]，不应崩溃且 loss=0（无有效位置）"""
        model = GPT(tiny_config)
        model.eval()

        input_ids = torch.tensor([[5]])
        targets = torch.tensor([[7]])
        with torch.no_grad():
            _, loss, _ = model(input_ids, targets)
        assert loss is not None
        assert torch.isfinite(loss).item()
        assert loss.item() == 0.0

    def test_forward_all_ignore_index_safe(self, tiny_config):
        """labels 全为 -100 时 loss 应安全返回 0，不 NaN"""
        model = GPT(tiny_config)
        model.eval()

        input_ids = torch.randint(0, tiny_config.vocab_size, (2, 8))
        targets = torch.full((2, 8), -100, dtype=torch.long)
        with torch.no_grad():
            _, loss, _ = model(input_ids, targets)
        assert torch.isfinite(loss).item()
        assert loss.item() == 0.0

    def test_gradient_flow(self, tiny_config, sample_input_ids):
        """梯度应正确回传"""
        model = GPT(tiny_config)
        model.train()

        targets = sample_input_ids.clone()
        logits, loss, _ = model(sample_input_ids, targets)
        loss.backward()

        # 检查至少有一个参数有梯度
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
        )
        assert has_grad, "反向传播应该产生梯度"

    def test_count_parameters(self, tiny_model, tiny_config):
        """模型实际参数量应与 config 估算一致 (数量级相同)"""
        actual = tiny_model.count_parameters()["total"]
        estimated = tiny_config.count_params()["total"]
        ratio = actual / estimated
        # 允许 50% 的差异 (weight tying 等)
        assert 0.5 < ratio < 2.0, (
            f"参数量差异过大: actual={actual}, estimated={estimated}"
        )

    def test_generate(self, tiny_model, tiny_config):
        """generate 应产生正确长度的序列"""
        from inference.generate import generate

        prompt = torch.randint(0, tiny_config.vocab_size, (1, 4))
        max_new = 8

        with torch.no_grad():
            output = generate(
                tiny_model, prompt, max_new_tokens=max_new, temperature=1.0, eos_token_id=-1
            )

        # 输出长度应至少包含 prompt 长度
        assert output.shape[0] == 1
        assert output.shape[1] >= 4
        assert output.shape[1] <= 4 + max_new

    def test_kv_cache_equivalence(self, tiny_model, tiny_config):
        """KV Cache 和无 Cache 的结果应一致"""
        prompt = torch.randint(0, tiny_config.vocab_size, (1, 8))

        with torch.no_grad():
            # 无 cache
            logits_no_cache, _, _ = tiny_model(prompt)
            last_no_cache = logits_no_cache[:, -1, :]

            # 有 cache: prefill 全部 token
            logits_cache, _, kv_caches = tiny_model(prompt, use_cache=True)
            last_cache = logits_cache[:, -1, :]

        # 最后一个 token 的 logits 应近似相等
        assert torch.allclose(last_no_cache, last_cache, atol=1e-4), (
            "KV Cache 的结果应与无 Cache 一致"
        )
