import sys, time
sys.path.insert(0, '.')
import torch

print('Building tiny FSTB model (d_model=128)...', flush=True)
from fstb.config import ModelConfig, MemoryConfig
from fstb.models.fstb_transformer import FSTBTransformer

cfg = ModelConfig(vocab_size=200, d_model=128, n_layers=24, n_heads=4, d_ff=256, max_seq_len=32)
mem = MemoryConfig(d_mem=64, d_sym=32)
m = FSTBTransformer(cfg, mem)
params = sum(p.numel() for p in m.parameters())
print(f'  Parameters: {params:,}', flush=True)

t0 = time.time()
x = torch.randint(0, 200, (2, 16))
out = m(x)
elapsed = time.time() - t0

logit_shape = out["logits"].shape
routing_shape = out["routing_probs"].shape
stage_a_keys = list(out["stage_a"].keys())

print(f'  Forward pass: {elapsed:.2f}s  logits={logit_shape}', flush=True)
print(f'  stage_a keys: {stage_a_keys}', flush=True)
print(f'  routing_probs shape: {routing_shape}', flush=True)
print('SUCCESS', flush=True)
