import unittest
import torch
from fstb.models.memory_interface import MemoryObject, MemoryType
from fstb.models.memory_controller import DynamicMemoryController

class TestMemoryController(unittest.TestCase):
    def setUp(self):
        self.controller = DynamicMemoryController(d_mem=32, d_sym=16, max_slots=10)

    def test_store_and_retrieve(self):
        mem1 = MemoryObject(
            memory_id="mem_1",
            content_embedding=torch.randn(32),
            symbolic_summary=torch.randn(16),
            memory_type=MemoryType.PERSISTENT_USER,
            importance=0.9,
            confidence=0.95
        )
        stored_id = self.controller.store(mem1)
        self.assertEqual(stored_id, "mem_1")
        
        results = self.controller.retrieve(mem1.content_embedding, top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0].memory_id, "mem_1")

    def test_update_and_invalidate(self):
        mem1 = MemoryObject(
            memory_id="mem_1",
            content_embedding=torch.randn(32),
            symbolic_summary=torch.randn(16)
        )
        self.controller.store(mem1)
        self.assertTrue(self.controller.update("mem_1", new_confidence=0.2))
        self.assertEqual(self.controller.memories["mem_1"].confidence, 0.2)
        
        self.assertTrue(self.controller.invalidate("mem_1"))
        self.assertNotIn("mem_1", self.controller.memories)

    def test_merge(self):
        mem1 = MemoryObject(memory_id="mem_1", content_embedding=torch.ones(32), symbolic_summary=torch.ones(16))
        mem2 = MemoryObject(memory_id="mem_2", content_embedding=torch.ones(32)*3, symbolic_summary=torch.ones(16)*3)
        self.controller.store(mem1)
        self.controller.store(mem2)

        merged = self.controller.merge("mem_1", "mem_2", "merged_1")
        self.assertIsNotNone(merged)
        self.assertIn("merged_1", self.controller.memories)
        self.assertNotIn("mem_1", self.controller.memories)
        self.assertNotIn("mem_2", self.controller.memories)
        self.assertTrue(torch.allclose(merged.content_embedding, torch.ones(32)*2))

if __name__ == "__main__":
    unittest.main()
