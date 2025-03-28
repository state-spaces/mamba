from typing import Any

import pytest
import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Replicate

from dtest import DTest
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.modules.moe import MoE


def _copy_params(model: nn.Module, model_fsdp: nn.Module) -> None:
    for n, m_fsdp in model_fsdp.named_modules():
        m = model.get_submodule(n)
        with torch.no_grad():
            for p_dest, p_src in zip(
                m_fsdp.parameters(recurse=False), m.parameters(recurse=False)
            ):
                p_dest.data.copy_(p_src.data)


def _test_grads(
    model: nn.Module, model_fsdp: nn.Module, tol: float, mesh: DeviceMesh
) -> None:
    for n, m_fsdp in model_fsdp.named_modules():
        m = model.get_submodule(n)
        for (n, p), (_, p_fsdp) in zip(
            m.named_parameters(recurse=False),
            m_fsdp.named_parameters(recurse=False),
        ):
            if p.grad is None:
                assert p_fsdp.grad is None
                return
            grad = p.grad
            grad_fsdp = p_fsdp.grad
            if isinstance(grad_fsdp, DTensor):
                grad_fsdp = grad_fsdp.redistribute(
                    mesh, placements=[Replicate() for _ in grad_fsdp.placements]
                ).to_local()
            try:
                torch.testing.assert_close(grad, grad_fsdp, atol=tol, rtol=tol)
            except Exception as e:
                raise RuntimeError(f"Failed on {n=}") from e


class _TestBase(DTest):
    in_features = 256
    hidden_features = in_features // 2
    n_shared_experts = 1
    n_activated_experts = 2
    n_layer = 2
    vocab_size = 512
    tie_embeddings = False

    seqlen = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Uniform FSDP2 dtype issues with bfloat32 b/c the dt_bias doesn't respect factory_kwargs
    dtype = torch.float32
    factory_kwargs = {"device": device, "dtype": dtype}
    ssm_cfg = {"layer": "Mamba2"}
    attn_layer_idx = [n_layer - 1]
    head_dim = 64
    attn_cfg = {
        "causal": True,
        "d_conv": 0,
        "head_dim": head_dim,
        "num_heads": 8,
        "num_heads_kv": 2,
        "out_proj_bias": False,
        "qkv_proj_bias": False,
        "rotary_emb_dim": head_dim // 2,
    }
    moe_layer_idx = list(range(1, n_layer))

    tol = 1e-2

    @property
    def n_routed_experts(self) -> int:
        return 2 * self.world_size

    @property
    def batch_size(self) -> int:
        return self.world_size

    @property
    def moe_cfg(self) -> int:
        return {
            "n_routed_experts": self.n_routed_experts,
            "n_activated_experts": 1,
            "n_shared_experts": 1,
        }

    @property
    def cfg(self) -> int:
        return MambaConfig(
            d_model=self.in_features,
            d_intermediate=self.hidden_features,
            n_layer=self.n_layer,
            vocab_size=self.vocab_size,
            tie_embeddings=self.tie_embeddings,
            attn_layer_idx=self.attn_layer_idx,
            attn_cfg=self.attn_cfg,
            moe_layer_idx=self.moe_layer_idx,
            moe_cfg=self.moe_cfg,
            ssm_cfg=self.ssm_cfg,
        )

    @property
    def factory_kwargs(self) -> dict[str, Any]:
        return {"device": self.device, "dtype": self.dtype}

    def get_inputs(self) -> torch.Tensor:
        return torch.randn(
            self.batch_size, self.seqlen, self.in_features, **self.factory_kwargs
        )

    def get_input_toks(self) -> torch.Tensor:
        return torch.randint(
            self.vocab_size, size=(self.batch_size, self.seqlen), device=self.device
        )


class TestMoEEP(_TestBase):
    @pytest.mark.world_size(4)
    @pytest.mark.gpu
    def test_fwd(self) -> None:
        torch.manual_seed(42)
        ep_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("ep",)
        )
        model_kwargs = dict(
            in_features=self.in_features,
            hidden_features=self.hidden_features,
            n_routed_experts=self.n_routed_experts,
            n_activated_experts=self.n_activated_experts,
            n_shared_experts=self.n_shared_experts,
            score_func="sigmoid",
            **self.factory_kwargs,
        )
        model = MoE(**model_kwargs)
        model_ep = MoE(**model_kwargs, ep_mesh=ep_mesh)

        # Force models equal
        _copy_params(model, model_ep)

        torch.manual_seed(42 + self.rank)
        inputs = self.get_inputs()
        outputs = model(inputs)
        outputs_ep = model_ep(inputs)

        torch.testing.assert_close(outputs, outputs_ep, atol=self.tol, rtol=self.tol)

    @pytest.mark.world_size(4)
    @pytest.mark.gpu
    def test_bwd(self) -> None:
        torch.manual_seed(42)
        ep_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("ep",)
        )
        model_kwargs = dict(
            in_features=self.in_features,
            hidden_features=self.hidden_features,
            n_routed_experts=self.n_routed_experts,
            n_activated_experts=self.n_activated_experts,
            n_shared_experts=self.n_shared_experts,
            score_func="sigmoid",
            **self.factory_kwargs,
        )
        model = MoE(**model_kwargs)
        model_ep = MoE(**model_kwargs, ep_mesh=ep_mesh)

        # Force models equal
        _copy_params(model, model_ep)

        fully_shard(model_ep.gate, mesh=ep_mesh)
        if model_ep.shared_experts is not None:
            fully_shard(model_ep.shared_experts, mesh=ep_mesh)

        # The ignored_params arg requires torch nightly (> 2.6.0)
        fully_shard(
            model_ep, mesh=ep_mesh, ignored_params=set(model_ep.experts.parameters())
        )

        inputs = self.get_inputs()
        outputs = model(inputs)

        inputs_ep = inputs.tensor_split(self.world_size, dim=0)[self.rank]
        outputs_ep = model_ep(inputs_ep)

        # Grads should match with an aver-over-batches type loss
        outputs.pow(2).mean().backward()
        outputs_ep.pow(2).mean().backward()

        _test_grads(model, model_ep, tol=self.tol, mesh=ep_mesh)
        # Verify the routed experts are not sharded and everything else is
        try:
            for n, p in model_ep.named_parameters():
                if n.startswith("experts"):
                    assert not isinstance(p, DTensor)
                else:
                    assert isinstance(p, DTensor)
        except Exception as e:
            raise RuntimeError(f"Failed on {n=}, {p=}") from e


class TestModelEP(_TestBase):
    @pytest.mark.world_size(4)
    @pytest.mark.gpu
    def test_fwd(self) -> None:
        torch.manual_seed(42)
        ep_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("ep",)
        )
        model = MambaLMHeadModel(self.cfg, **self.factory_kwargs)
        model_ep = MambaLMHeadModel(self.cfg, **self.factory_kwargs, ep_mesh=ep_mesh)

        # Verify EP
        print(f"{model_ep=}")
        for m, m_ep in zip(model.modules(), model_ep.modules()):
            if isinstance(m, MoE):
                assert len(m_ep.experts) == len(m.experts) // self.world_size

        # Force models equal
        _copy_params(model, model_ep)

        torch.manual_seed(42 + self.rank)
        inputs = self.get_input_toks()
        outputs = model(inputs)
        outputs_ep = model_ep(inputs)

        torch.testing.assert_close(outputs, outputs_ep, atol=self.tol, rtol=self.tol)


    @pytest.mark.world_size(4)
    @pytest.mark.gpu
    def test_bwd(self) -> None:
        torch.manual_seed(42)
        ep_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("ep",)
        )
        model = MambaLMHeadModel(self.cfg, **self.factory_kwargs)
        model_ep = MambaLMHeadModel(self.cfg, **self.factory_kwargs, ep_mesh=ep_mesh)

        # Force models equal
        _copy_params(model, model_ep)

        fully_shard(model_ep.lm_head, mesh=ep_mesh)
        fully_shard(model_ep.backbone.embedding, mesh=ep_mesh)
        for block in model_ep.backbone.layers.values():
            # The ignored_params arg requires torch nightly (> 2.6.0)
            ignored_params = (
                set(block.mlp.experts.parameters())
                if isinstance(block.mlp, MoE)
                else None
            )
            fully_shard(block, mesh=ep_mesh, ignored_params=ignored_params)

        fully_shard(model_ep, mesh=ep_mesh)

        inputs = self.get_input_toks()
        outputs = model(inputs)

        inputs_ep = inputs.tensor_split(self.world_size, dim=0)[self.rank]
        outputs_ep = model_ep(inputs_ep)

        # Grads should match with an aver-over-batches type loss
        outputs.logits.pow(2).mean().backward()
        outputs_ep.logits.pow(2).mean().backward()

        _test_grads(model, model_ep, tol=self.tol, mesh=ep_mesh)

        # Verify the routed experts are not sharded and everything else is
        try:
            for n, p in model_ep.named_parameters():
                if ".experts." in n:
                    assert not isinstance(p, DTensor)
                else:
                    assert isinstance(p, DTensor)
        except Exception as e:
            raise RuntimeError(f"Failed on {n=}, {p=}") from e
