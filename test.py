import torch
from fstb.models.model_factory import build_model_trio, verify_parameter_parity

def test():
    print("Building tiny trio...")
    baseline, aux, fstb = build_model_trio("tiny")
    
    parity = verify_parameter_parity(baseline, aux, fstb)
    print("Parameter parity:", parity)
    
    # Test forward pass
    print("Testing forward passes...")
    input_ids = torch.randint(0, 4096, (2, 64))
    
    b_out = baseline(input_ids)
    print("Baseline logits shape:", b_out["logits"].shape)
    
    a_out = aux(input_ids)
    print("AuxBaseline logits shape:", a_out["logits"].shape)
    print("AuxBaseline stage A keys:", list(a_out["stage_a"].keys()))
    
    f_out = fstb(input_ids, block_ablation_mode='zero_stage_b')
    print("FSTB logits shape:", f_out["logits"].shape)
    print("FSTB stage B keys:", list(f_out["stage_b"].keys()))
    
    print("All tests passed!")

if __name__ == "__main__":
    test()
