"""FP64 regression tests for phase precision and long-sequence angle gradients."""
import math
import pytest
import torch
import torch.nn.functional as F

NAMES = ('Q', 'K', 'V', 'ADT', 'DT', 'Trap', 'Q_bias', 'K_bias', 'Angles')


def inputs(length, heads=4, groups=1, width=64, value=64, batch=1, device='cuda', seed=101, rates_fp32=False):
    torch.manual_seed(seed)
    dtype = torch.bfloat16 if device == 'cuda' else torch.float64
    def rand(*s):
        return torch.randn(*s, device=device, dtype=dtype)
    q, k = [F.rms_norm(rand(batch, length, groups, width), (width,)) for _ in range(2)]
    dt = torch.exp(torch.empty(batch, heads, length, device=device).uniform_(math.log(.001), math.log(.1)))
    adt = -F.softplus(torch.randn_like(dt)) * dt
    angles = rand(batch, length, heads, width // 4).float()
    if rates_fp32:
        angles = torch.randn_like(angles)
    return dict(zip(NAMES, [q, k, rand(batch, length, heads, value), adt, dt,
                           rand(batch, heads, length), rand(heads, width).float(),
                           rand(heads, width).float(), angles]))


def leaves(data, dtype=None):
    return {k: v.detach().to(dtype or v.dtype).clone().requires_grad_() for k, v in data.items()}


def rotated(data):
    q, k, v, adt, dt, trap, qb, kb, angles = [data[n] for n in NAMES]
    h = v.shape[2]
    q = q.repeat_interleave(h // q.shape[2], 2) + qb
    k = k.repeat_interleave(h // k.shape[2], 2) + kb
    theta = (angles.tanh() * math.pi * dt.transpose(1, 2)[..., None]).cumsum(1)
    r = theta.shape[-1] * 2
    def rot(x):
        a, b = x[..., :r:2], x[..., 1:r:2]
        c, s = theta.cos(), theta.sin()
        return torch.cat((torch.stack((a*c-b*s, a*s+b*c), -1).flatten(-2), x[..., r:]), -1)
    return rot(q), rot(k), v, adt, dt, trap.sigmoid(), theta, (q*k).sum(-1)


def reference(data, token=False, chunk=64):
    q, k, v, adt, dt, trap, theta, qk = rotated(data)
    b, length, h, n = q.shape
    p = v.shape[-1]
    state = q.new_zeros(b, h, n, p)
    outputs = []
    if token:
        pk, pv = torch.zeros_like(k[:, 0]), torch.zeros_like(v[:, 0])
        for t in range(length):
            a = adt[:, :, t].exp()[..., None, None]
            old = (dt[:, :, t]*(1-trap[:, :, t]))[..., None, None]
            new = (dt[:, :, t]*trap[:, :, t])[..., None, None]
            state = a*(state + old*pk[..., :, None]*pv[..., None, :]) + new*k[:, t, ..., :, None]*v[:, t, ..., None, :]
            outputs.append(torch.einsum('bhn,bhnp->bhp', q[:, t], state))
            pk, pv = k[:, t], v[:, t]
        return torch.stack(outputs, 1)
    shift = F.pad((dt*(1-trap))[..., 1:], (0, 1))
    scale = dt*trap + shift
    for start in range(0, length, chunk):
        stop = min(start+chunk, length)
        qc = q[:, start:stop].transpose(1, 2)
        kc = k[:, start:stop].transpose(1, 2) * scale[:, :, start:stop, None]
        vc = v[:, start:stop].transpose(1, 2)
        cs = adt[:, :, start:stop].cumsum(-1)
        diff = cs[..., :, None] - cs[..., None, :]
        mask = torch.ones(stop-start, stop-start, device=q.device, dtype=torch.bool).tril()
        decay = diff.masked_fill(~mask, -torch.inf).exp()
        out = (qc @ kc.transpose(-1, -2) * decay) @ vc
        out = out + (qc @ state)*cs.exp()[..., None]
        outputs.append(out.transpose(1, 2))
        state = state*cs[..., -1, None, None].exp() + kc.transpose(-1, -2) @ (vc*(cs[..., -1, None]-cs).exp()[..., None])
    return torch.cat(outputs, 1) - v*(qk*shift.transpose(1, 2))[..., None]


def cpu_check():
    data = inputs(17, heads=2, width=8, value=4, device='cpu')
    a, b = leaves(data, torch.float64), leaves(data, torch.float64)
    x, y = reference(a, chunk=5), reference(b, token=True)
    cot = torch.randn_like(x)
    ga = torch.autograd.grad(x, tuple(a.values()), cot)
    gb = torch.autograd.grad(y, tuple(b.values()), cot)
    torch.testing.assert_close(x, y, rtol=1e-11, atol=1e-11)
    for u, v in zip(ga, gb):
        torch.testing.assert_close(u, v, rtol=1e-10, atol=1e-10)
    print('CPU_FP64_ORACLE_OUTPUT_AND_ALL_GRADIENTS_PASS', flush=True)


def test_fp64_oracle_cpu():
    cpu_check()


@pytest.mark.parametrize('length,seed', [(73, 101), (4096, 1111), (8192, 2222)])
def test_angle_gradient_fp64(length, seed):
    if not torch.cuda.is_available():
        pytest.skip('requires CUDA')
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined
    data = inputs(length, seed=seed, rates_fp32=True)
    a, b = leaves(data), leaves(data, torch.float64)
    expected = reference(b)
    actual = mamba3_siso_combined(**a, chunk_size=64)
    cot = torch.randn_like(expected).bfloat16()
    ga = torch.autograd.grad(actual, tuple(a.values()), cot.to(actual.dtype))
    gb = torch.autograd.grad(expected, tuple(b.values()), cot.double())
    errors = {}
    for name, got, ref in [('output', actual, expected), *zip(NAMES, ga, gb)]:
        relative = ((got.double() - ref).norm() / ref.norm().clamp_min(1e-20)).item()
        errors[name] = relative
        assert torch.isfinite(got).all(), name
        assert relative < (0.015 if name == 'Angles' else 0.02), (name, relative)
    print(length, seed, errors)


def test_phase_storage_and_expanded_rates():
    if not torch.cuda.is_available():
        pytest.skip('requires CUDA')
    from mamba_ssm.ops.triton.mamba3.angle_dt import angle_dt_fwd
    torch.manual_seed(101)
    rates = torch.randn(1, 8192, 1, 16, device='cuda').expand(-1, -1, 4, -1)
    dt = torch.rand(1, 4, 8192, device='cuda') * 0.1
    expected = (rates.double().tanh() * math.pi * dt.double().transpose(1, 2)[..., None]).cumsum(1)
    phases, state = angle_dt_fwd(rates, dt, return_output_state=True)
    assert phases.dtype == state.dtype == torch.float32
    diff = (phases.double() - expected + math.pi).remainder(2 * math.pi) - math.pi
    assert diff.square().mean().sqrt() < 1e-4
    bf16_phases, bf16_state = angle_dt_fwd(rates.bfloat16(), dt, return_output_state=True)
    assert bf16_phases.dtype == bf16_state.dtype == torch.float32
