"""
HuggingFace Pretrained Weight Loader for GPT-2.

Maps HuggingFace GPT2LMHeadModel state dict keys to our GPT model.
Note: HF GPT-2 uses Conv1D (transposed weights) — we transpose during loading.
"""

import torch
from typing import Optional


def load_gpt2_weights(model, hf_name: str = "gpt2", device: Optional[str] = None):
    """
    Load pretrained HuggingFace GPT-2 weights into our GPT model.

    Args:
        model: Our GPT model instance
        hf_name: HuggingFace model name ('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl')
        device: Device to load weights to

    Returns:
        model with loaded weights
    """
    from transformers import GPT2LMHeadModel

    print(f"  Loading pretrained weights from HuggingFace: {hf_name}")
    hf_model = GPT2LMHeadModel.from_pretrained(hf_name)
    hf_sd = hf_model.state_dict()

    # Build mapping from HF keys to our keys
    sd = model.state_dict()
    mapped = {}

    # Token and position embeddings
    mapped['token_embedding.weight'] = hf_sd['transformer.wte.weight']
    mapped['position_embedding.weight'] = hf_sd['transformer.wpe.weight']

    # Transformer blocks
    for i in range(model.num_layers):
        hf_prefix = f'transformer.h.{i}'
        our_prefix = f'blocks.{i}'

        # Layer norms
        mapped[f'{our_prefix}.ln1.weight'] = hf_sd[f'{hf_prefix}.ln_1.weight']
        mapped[f'{our_prefix}.ln1.bias'] = hf_sd[f'{hf_prefix}.ln_1.bias']
        mapped[f'{our_prefix}.ln2.weight'] = hf_sd[f'{hf_prefix}.ln_2.weight']
        mapped[f'{our_prefix}.ln2.bias'] = hf_sd[f'{hf_prefix}.ln_2.bias']

        # Attention: HF uses a single Conv1D for c_attn (QKV combined)
        # Conv1D stores weight as [in_features, out_features] (transposed vs Linear)
        c_attn_weight = hf_sd[f'{hf_prefix}.attn.c_attn.weight']  # [d_model, 3*d_model]
        c_attn_bias = hf_sd[f'{hf_prefix}.attn.c_attn.bias']      # [3*d_model]

        d_model = model.d_model
        # Split into Q, K, V — each [d_model, d_model]
        # HF Conv1D: weight is [in, out], nn.Linear expects [out, in]
        q_w, k_w, v_w = c_attn_weight.split(d_model, dim=1)
        q_b, k_b, v_b = c_attn_bias.split(d_model, dim=0)

        # nn.MultiheadAttention stores in_proj_weight as [3*d_model, d_model]
        mapped[f'{our_prefix}.attn.in_proj_weight'] = torch.cat([q_w.T, k_w.T, v_w.T], dim=0)
        mapped[f'{our_prefix}.attn.in_proj_bias'] = torch.cat([q_b, k_b, v_b], dim=0)

        # Attention output projection: HF c_proj is Conv1D
        c_proj_weight = hf_sd[f'{hf_prefix}.attn.c_proj.weight']  # [d_model, d_model]
        c_proj_bias = hf_sd[f'{hf_prefix}.attn.c_proj.bias']
        mapped[f'{our_prefix}.attn.out_proj.weight'] = c_proj_weight.T
        mapped[f'{our_prefix}.attn.out_proj.bias'] = c_proj_bias

        # FFN: HF uses Conv1D for mlp.c_fc and mlp.c_proj
        # Our FFN: Sequential(Linear, GELU, Linear, Dropout)
        # ffn[0] = first Linear, ffn[2] = second Linear
        mlp_fc_weight = hf_sd[f'{hf_prefix}.mlp.c_fc.weight']    # [d_model, d_ff]
        mlp_fc_bias = hf_sd[f'{hf_prefix}.mlp.c_fc.bias']
        mlp_proj_weight = hf_sd[f'{hf_prefix}.mlp.c_proj.weight']  # [d_ff, d_model]
        mlp_proj_bias = hf_sd[f'{hf_prefix}.mlp.c_proj.bias']

        mapped[f'{our_prefix}.ffn.0.weight'] = mlp_fc_weight.T
        mapped[f'{our_prefix}.ffn.0.bias'] = mlp_fc_bias
        mapped[f'{our_prefix}.ffn.2.weight'] = mlp_proj_weight.T
        mapped[f'{our_prefix}.ffn.2.bias'] = mlp_proj_bias

    # Final layer norm
    mapped['final_norm.weight'] = hf_sd['transformer.ln_f.weight']
    mapped['final_norm.bias'] = hf_sd['transformer.ln_f.bias']

    # LM head is weight-tied with token_embedding, so we don't load it separately.
    # But verify the shapes match.
    assert mapped['token_embedding.weight'].shape == sd['token_embedding.weight'].shape, \
        f"Token embedding shape mismatch: {mapped['token_embedding.weight'].shape} vs {sd['token_embedding.weight'].shape}"

    # Load mapped weights
    missing, unexpected = [], []
    for key in sd:
        if key in mapped:
            if sd[key].shape != mapped[key].shape:
                print(f"  Warning: Shape mismatch for {key}: "
                      f"model={sd[key].shape}, pretrained={mapped[key].shape}")
                unexpected.append(key)
            else:
                sd[key] = mapped[key]
        elif key == 'lm_head.weight':
            # Weight-tied, skip
            pass
        else:
            missing.append(key)

    model.load_state_dict(sd, strict=False)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {len(mapped)} weight tensors ({n_params:,} parameters)")
    if missing:
        print(f"  Missing keys (not in HF model): {missing}")
    if unexpected:
        print(f"  Shape mismatches: {unexpected}")

    return model
