import unittest
import torch
from fstb.config import ModelConfig, MemoryConfig, LossWeightsConfig
from fstb.models.baseline import BaselineTransformer
from fstb.models.fstb_transformer import FSTBTransformer
from fstb.models.loss import FSTBLossFunction

class TestFSTBModels(unittest.TestCase):
    def setUp(self):
        self.model_cfg = ModelConfig(vocab_size=100, d_model=64, n_layers=24, n_heads=4, d_ff=128, max_seq_len=32)
        self.mem_cfg = MemoryConfig(d_mem=32, d_sym=16)

    def test_baseline_forward(self):
        model = BaselineTransformer(self.model_cfg)
        input_ids = torch.randint(0, 100, (2, 16))
        outputs = model(input_ids, return_hidden_states=True)
        self.assertIn("logits", outputs)
        self.assertEqual(outputs["logits"].shape, (2, 16, 100))
        self.assertEqual(len(outputs["hidden_states"]), 24)

    def test_fstb_forward(self):
        model = FSTBTransformer(self.model_cfg, self.mem_cfg)
        input_ids = torch.randint(0, 100, (2, 16))
        outputs = model(input_ids, return_hidden_states=True)
        self.assertIn("logits", outputs)
        self.assertEqual(outputs["logits"].shape, (2, 16, 100))
        self.assertIn("stage_a", outputs)
        self.assertIn("stage_b", outputs)
        self.assertIn("stage_c", outputs)
        self.assertEqual(len(outputs["hidden_states"]), 24)

    def test_fstb_loss(self):
        loss_fn = FSTBLossFunction(LossWeightsConfig())
        model = FSTBTransformer(self.model_cfg, self.mem_cfg)
        input_ids = torch.randint(0, 100, (2, 16))
        outputs = model(input_ids)
        targets = {
            "input_ids": input_ids,
            "importance_target": torch.rand(2, 16),
            "memory_worthiness_target": torch.randint(0, 2, (2, 16)).float(),
            "contradiction_target": torch.randint(0, 2, (2, 16)).float()
        }
        loss_dict = loss_fn(outputs, targets)
        self.assertIn("loss", loss_dict)
        self.assertTrue(loss_dict["loss"].item() > 0.0)

if __name__ == "__main__":
    unittest.main()
