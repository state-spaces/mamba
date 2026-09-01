import torch

from mamba_ssm.modules.ssd_minimal import ssd_minimal_discrete


def test_initial_state_round_trip_matches_full_sequence():
    torch.manual_seed(42)

    batch, seqlen, nheads, dhead, dstate = 2, 12, 3, 2, 4
    block_len = 3
    split = 6
    dtype = torch.float64

    x = torch.randn(batch, seqlen, nheads, dhead, dtype=dtype)
    a = -torch.rand(batch, seqlen, nheads, dtype=dtype)
    b = torch.randn(batch, seqlen, nheads, dstate, dtype=dtype)
    c = torch.randn(batch, seqlen, nheads, dstate, dtype=dtype)

    y_full, final_state_full = ssd_minimal_discrete(x, a, b, c, block_len)
    y_first, intermediate_state = ssd_minimal_discrete(
        x[:, :split], a[:, :split], b[:, :split], c[:, :split], block_len
    )
    y_second, final_state_chunked = ssd_minimal_discrete(
        x[:, split:],
        a[:, split:],
        b[:, split:],
        c[:, split:],
        block_len,
        initial_states=intermediate_state,
    )

    assert intermediate_state.shape == (batch, nheads, dhead, dstate)
    torch.testing.assert_close(
        torch.cat([y_first, y_second], dim=1), y_full, rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(
        final_state_chunked, final_state_full, rtol=1e-12, atol=1e-12
    )
