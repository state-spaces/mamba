from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.utils.generation import InferenceParams

import torch
import copy


def test_state_seq():
    """Check that Mamba([x1.x2.x3.x4]) == Mamba([x1,x2])|>step(x3)|>step(x4)"""
    device = "cuda:0"
    dim_model = 8

    # Generate a model with random weights
    m = Mamba(dim_model, layer_idx=7).to(device=device)
    m.requires_grad_(False)  # allows deepcopy of tensrors

    # Generate the whole sequence
    x_all = torch.rand(1, 5, dim_model, device=device)
    y_all = m(x_all)

    # Introducing empty inference parameters should not add new data
    inference_all = InferenceParams(max_seqlen=16, max_batch_size=3)
    y_with_inference = m(x_all, inference_params=inference_all)

    assert len(inference_all.key_value_memory_dict)
    assert torch.allclose(y_with_inference, y_all)

    # Inference by parts
    # X0,X1
    inference_part = InferenceParams(
        max_seqlen=inference_all.max_seqlen, max_batch_size=inference_all.max_batch_size)
    y01 = m(x_all[:, 0:2], inference_params=inference_part)
    assert torch.allclose(y_with_inference[:, :2], y01)

    # (past state up to X1), X2, X3
    inference_part.seqlen_offset = 2
    inference_part_b = copy.deepcopy(inference_part)
    y2 = m(x_all[:, 2:4], inference_params=inference_part)

    # (past state up to X3), X4
    inference_part.seqlen_offset = 4
    y3 = m(x_all[:, 4:5], inference_params=inference_part)

    # (past state up to X1), X2 again
    inference_part_b.seqlen_offset = 2
    y2_b = m(x_all[:, 2:3], inference_params=inference_part_b)
    # (past state up to X2), X3 again
    inference_part_b.seqlen_offset = 3
    y3_b = m(x_all[:, 3:4], inference_params=inference_part_b)

    # Values should match result we got from inferencin over the all sequence
    assert torch.allclose(y_all[:, 0:2], y01)
    assert torch.allclose(y_all[:, 2:4], y2) #Decode chunk - Finally works.
    assert torch.allclose(y_all[:, 4:5], y3)
    assert torch.allclose(y_all[:, 2:3], y2_b)
    assert torch.allclose(y_all[:, 3:4], y3_b)

    # Sanity check
    assert not torch.allclose(y_all[:, 3:4], y2)


def test_state_batch_drop_empty_infer():
    """Check that you can drop a batch when inference parms are empty"""
    device = "cuda"
    dim_model = 8

    # Generate a model with random weights
    m = Mamba(dim_model, layer_idx=7).to(device=device)
    m.requires_grad_(False)  # allows deepcopy of tensrors

    x_all = torch.rand(3, 4, dim_model, device=device)
    y_all = m(x_all)

    # Introducing empty inference parameters should not add new data
    inference_all = InferenceParams(max_seqlen=16, max_batch_size=3)
    y_all = m(x_all, inference_params=inference_all)
    kv = inference_all.key_value_memory_dict[7]

    # Drop batch in the middle
    x_02 = x_all[(0, 2), ...]
    kv = tuple(batched[(0, 2), ...] for batched in kv)
    inference_all.key_value_memory_dict[7] = kv

    inference_02 = InferenceParams(max_seqlen=16, max_batch_size=3)
    y_02 = m(x_02, inference_params=inference_02)
    y_02_a = y_all[(0, 2), ...]
    assert torch.allclose(y_02, y_02_a)


def test_state_batch_drop_step():
    """Check that you can drop a batch when inference parms are filled"""

    device = "cuda"
    dim_model = 8

    # Generate a model with random weights
    m = Mamba(dim_model, layer_idx=7).to(device=device)
    m.requires_grad_(False)  # allows deepcopy of tensrors

    x_prefix = torch.rand(3, 4, dim_model, device=device)

    # Rewind model forward so inference parms has data
    inference_parms = InferenceParams(max_seqlen=16, max_batch_size=3)
    _ = m(x_prefix, inference_params=inference_parms)

    x_next = torch.rand(3, 1, dim_model, device=device)
    inference_parms.seqlen_offset = x_prefix.shape[1]
    inference_parms_bak = copy.deepcopy(inference_parms)

    # Y with all 3 batches
    y_next = m(x_next, inference_params=inference_parms)

    # Remove middle batch from cache
    kv = inference_parms_bak.key_value_memory_dict[7]
    kv = tuple(batched[(0, 2), ...] for batched in kv)
    inference_parms_bak.key_value_memory_dict[7] = kv

    # Calculate batches without middle batch
    x_02 = x_next[(0, 2), ...]
    y_next_parmed = m(x_02, inference_params=inference_parms_bak)

    # Check that batch was removed
    y_next_a = y_next[(0, 2), ...]
    assert torch.allclose(y_next_a, y_next_parmed)

    # Sanity check
    assert not torch.allclose(y_next[(0, 1), ...], y_next_parmed)


test_state_seq()
# test_state_batch_drop_empty_infer()
# test_state_batch_drop_step()