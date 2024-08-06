import torch
import torch.nn.functional as F

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.utils.generation import InferenceParams

import pytest

from einops import rearrange, repeat


def test_generation():
    batch = 3
    seqlen = 20
    device = "cuda"
    dtype = torch.float16

    config = MambaConfig(
        d_model=1024,
        n_layer=4,
        vocab_size=50277,
        ssm_cfg=dict(layer="Mamba2"),
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        pad_vocab_size_multiple=16,
    )
    torch.manual_seed(2357)
    model = MambaLMHeadModel(config, device=device, dtype=dtype)
    x = torch.randint(0, 1000, (batch, seqlen), device=device, dtype=torch.long)
    out_ref = model(x).logits
    prompt_len = seqlen // 2
    out = model.generate(
        input_ids = x[:, :prompt_len], max_length=seqlen, output_scores=True, return_dict_in_generate=True,
        cg=True,  # Can turn off CUDA graph for easier debugging
        # instead of sampling, we take output tokens from x, to get logits for testing
        # For actual generation, don't pass in teacher_outputs
        teacher_outputs=x,
    )
    out_scores = torch.stack(out.scores, dim=1)
    print(f"Max diff: {(out_scores - out_ref[:, prompt_len - 1: -1]).abs().max()}")
    assert torch.allclose(out_scores, out_ref[:, prompt_len - 1: -1], rtol=1e-3, atol=1e-2)


def test_generation_varlen():
    seqlens = [170, 65, 100]
    genlen = 20
    total_seqlen = sum(seqlens)
    device = "cuda"
    dtype = torch.float16

    config = MambaConfig(
        d_model=1024,
        n_layer=4,
        vocab_size=50277,
        ssm_cfg=dict(layer="Mamba2"),
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        pad_vocab_size_multiple=16,
    )
    torch.manual_seed(2357)
    model = MambaLMHeadModel(config, device=device, dtype=dtype)
    xs = [torch.randint(0, 1000, (1, seqlen), device=device, dtype=torch.long) for seqlen in seqlens]

    # Reference 1: Forward pass with seq_idx
    x = torch.cat(xs, dim=1)
    seq_idx = torch.cat([torch.full((ids.shape[1],), i, dtype=torch.int32, device=device)
                         for i, ids in enumerate(xs)], dim=0).unsqueeze(0)
    cu_seqlens = F.pad(torch.tensor(seqlens, device=device, dtype=torch.int32).cumsum(dim=0), (1, 0))
    out_ref = model(x, seq_idx=seq_idx).logits
    # Only take the last @genlen logits of each sequence
    out_ref = torch.cat([out_ref[:, cu_seqlens[i + 1] - genlen - 1:cu_seqlens[i + 1] - 1]
                         for i in range(len(seqlens))], dim=0)

    # Reference 2: Generate the last @genlen tokens of each sequence in a for loop
    out_loop = []
    for input_ids in xs:
        out = model.generate(
            input_ids=input_ids[:, :-genlen], max_length=input_ids.shape[1], output_scores=True,
            return_dict_in_generate=True, cg=True, teacher_outputs=input_ids,
        ).scores
        out_loop.append(torch.stack(out, dim=1))
    out_loop = torch.cat(out_loop, dim=0)
    print(f"Max diff between ref1 and ref2: {(out_loop - out_ref).abs().max()}")

    # Varlen generation
    input_ids = torch.cat([ids[:, :-genlen] for ids in xs], dim=1)
    prompt_seqlens = [seqlen - genlen for seqlen in seqlens]
    cu_seqlens = F.pad(torch.tensor(prompt_seqlens, device=device, dtype=torch.int32).cumsum(dim=0), (1, 0))
    seq_idx = torch.cat([torch.full((seqlen,), i, dtype=torch.int32, device=device)
                         for i, seqlen in enumerate(prompt_seqlens)], dim=0).unsqueeze(0)
    inference_params = InferenceParams(max_seqlen=2048, max_batch_size=len(seqlens))

    scores, sequences = [], []
    # Both seq_idx and cu_seqlens must be passed in for varlen generation
    logits = model(input_ids, inference_params=inference_params, seq_idx=seq_idx, cu_seqlens=cu_seqlens).logits
    logits = rearrange(logits[0, cu_seqlens[1:] - 1], "b d -> b 1 d")
    scores.append(logits)
    # In practice we should sample. In this case we take from the teacher_output for testing
    sampled_tokens = rearrange(torch.stack([ids[0, -genlen] for ids in xs], dim=0), "b -> b 1")
    sequences.append(sampled_tokens)
    for i in range(1, genlen):
        inference_params.seqlen_offset += 1
        logits = model(sampled_tokens, inference_params=inference_params, num_last_tokens=1).logits
        scores.append(logits)
        # In practice we should sample. In this case we take from the teacher_output for testing
        sampled_tokens = rearrange(torch.stack([ids[0, -genlen + i] for ids in xs], dim=0), "b -> b 1")
        sequences.append(sampled_tokens)
    out_varlen = torch.cat(scores, dim=1)
    print(f"Max diff: {(out_varlen - out_ref).abs().max()}")
    assert (out_varlen - out_ref).abs().max() < 2 * (out_loop - out_ref).abs().max()
