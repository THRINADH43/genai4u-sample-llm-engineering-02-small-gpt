import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model import GPT, CausalSelfAttention, GPTConfig, sample_next  # noqa: E402


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.cfg = GPTConfig(vocab_size=20, block_size=8, n_layer=2, n_head=2, n_embd=16, dropout=0.0)

    def test_attention_is_causal(self):
        """Changing a later token must not change the output at earlier positions."""
        attn = CausalSelfAttention(self.cfg).eval()
        x = torch.randn(1, 8, 16)
        y1 = attn(x)
        x2 = x.clone()
        x2[0, 5:] = torch.randn(3, 16)
        y2 = attn(x2)
        self.assertTrue(torch.allclose(y1[0, :5], y2[0, :5], atol=1e-6))
        self.assertFalse(torch.allclose(y1[0, 5:], y2[0, 5:]))

    def test_shapes_and_loss(self):
        model = GPT(self.cfg)
        idx = torch.randint(20, (3, 8))
        logits, loss = model(idx, idx)
        self.assertEqual(logits.shape, (3, 8, 20))
        self.assertAlmostEqual(loss.item(), torch.log(torch.tensor(20.0)).item(), delta=0.5)

    def test_can_overfit_one_batch(self):
        model = GPT(self.cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        x = torch.randint(20, (2, 8))
        y = torch.roll(x, -1, dims=1)
        for _ in range(150):
            _, loss = model(x, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.assertLess(loss.item(), 0.3)

    def test_too_long_sequence(self):
        with self.assertRaises(ValueError):
            GPT(self.cfg)(torch.zeros(1, 9, dtype=torch.long))


class SamplingTests(unittest.TestCase):
    def test_temperature_zero_is_greedy(self):
        logits = torch.tensor([[1.0, 5.0, 2.0]])
        self.assertEqual(sample_next(logits, temperature=0).item(), 1)

    def test_top_p_drops_unlikely_tokens(self):
        torch.manual_seed(0)
        # probabilities about 0.88, 0.12, 0.00...; top_p 0.5 keeps only the first
        logits = torch.tensor([[4.0, 2.0, -5.0, -5.0]])
        picks = {sample_next(logits, 1.0, 0.5).item() for _ in range(200)}
        self.assertEqual(picks, {0})

    def test_top_p_one_keeps_everything(self):
        torch.manual_seed(0)
        logits = torch.zeros(1, 4)
        picks = {sample_next(logits, 1.0, 1.0).item() for _ in range(400)}
        self.assertEqual(picks, {0, 1, 2, 3})


if __name__ == "__main__":
    unittest.main()
