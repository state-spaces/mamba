from typing import Literal, Optional

import torch
import torch.distributed._functional_collectives as funcol
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from mamba_ssm.modules.mlp import GatedMLP


class Gate(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

    Attributes:
        in_features (int): Dimensionality of input features.
        n_routed_experts (int): Number of experts available for token routing.
        n_activated_experts (int): Number of top experts activated for each input.
        score_func (str): Scoring function ('softmax' or 'sigmoid').
        route_scale (float): Scaling factor for routing weights.
        weight (torch.nn.Parameter): Learnable weights for the gate.
        bias (Optional[torch.nn.Parameter]): Optional bias term for the gate.
    """

    def __init__(
        self,
        in_features: int,
        n_routed_experts: int,
        n_activated_experts: int,
        score_func: Literal["sigmoid", "softmax"] = "softmax",
        route_scale: float = 1.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.n_routed_experts = n_routed_experts
        self.n_activated_experts = n_activated_experts
        self.score_func = score_func
        self.route_scale = route_scale
        factory_kwargs = {"device": device, "dtype": dtype}

        self.lin = nn.Linear(
            self.in_features, self.n_routed_experts, bias=False, **factory_kwargs
        )
        # Fix bias usage
        self.bias = (
            nn.Parameter(torch.empty(self.n_routed_experts, **factory_kwargs))
            if self.in_features == 7168
            else None
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        scores = self.lin(x)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            raise ValueError(f"Unexpected {self.score_func=} not in (softmax, sigmoid)")
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        indices = torch.topk(scores, self.n_activated_experts, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights.type_as(x), indices


class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        in_features (int): Dimensionality of input features.
        hidden_features (int): Dimensionality of hidden features of each expert.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
        ep_mesh (Optional[DeviceMesh]): 1D device mesh for expert parallel, if desired.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        n_routed_experts: int,
        n_shared_experts: int,
        n_activated_experts: int,
        score_func: Literal["sigmoid", "softmax"] = "softmax",
        route_scale: float = 1.0,
        ep_mesh: Optional[DeviceMesh] = None,
        device=None,
        dtype=None,
    ):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        if ep_mesh is not None and n_routed_experts % ep_mesh.size():
            raise ValueError(
                f"{self.n_routed_experts=} must be divisible by {ep_mesh.size()=}"
            )
        if ep_mesh is not None and ep_mesh.ndim != 1:
            raise ValueError(
                f"The expert parallel mesh must be one-dimensional: {ep_mesh.ndim=}"
            )

        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.score_func = score_func
        self.route_scale = route_scale
        self.ep_mesh = ep_mesh
        self.n_activated_experts = n_activated_experts

        factory_kwargs = {"device": device, "dtype": dtype}

        self.ep_mesh_size = 1 if ep_mesh is None else ep_mesh.size()
        self.n_local_experts = self.n_routed_experts // (
            self.ep_mesh.size() if self.ep_mesh is not None else 1
        )

        self.experts_start_idx = (
            0 if ep_mesh is None else ep_mesh.get_local_rank() * self.n_local_experts
        )
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(
            in_features=self.in_features,
            n_routed_experts=self.n_routed_experts,
            n_activated_experts=self.n_activated_experts,
            score_func=self.score_func,
            route_scale=self.route_scale,
            **factory_kwargs,
        )
        self.experts = nn.ModuleDict(
            {
                str(i): GatedMLP(
                    self.in_features, self.hidden_features, **factory_kwargs
                )
                for i in range(self.experts_start_idx, self.experts_end_idx)
            }
        )
        self.shared_experts = (
            GatedMLP(
                self.in_features,
                self.n_shared_experts * self.hidden_features,
                **factory_kwargs,
            )
            if self.n_shared_experts
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """

        x_shape = x.size()
        x = x.view(-1, self.in_features)

        weights, indices = self.gate(x)
        # counts[e] = num tokens this rank sends to expert e
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts)

        if self.ep_mesh is None:
            z = self._get_routed_expert_outputs(x, weights, indices, counts)
        else:
            z = self._get_ep_routed_expert_outputs(x, weights, indices, counts)

        if self.shared_experts is None:
            return z.view(x_shape)

        return (z + self.shared_experts(x)).view(x_shape)

    def _get_routed_expert_outputs(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        z = torch.zeros_like(x)
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[str(i)]
            idx, top = torch.where(indices == i)
            z[idx] += expert(x[idx]) * weights[idx, top, None]
        return z

    def _get_ep_routed_expert_outputs(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        # Sort tokens by the expert they are indexed to.
        flat_sorted_indices = indices.flatten().argsort(dim=-1)
        x_by_expert = x[flat_sorted_indices // self.n_activated_experts]

        assert self.ep_mesh is not None  # mypy
        # Get counts of incoming tensors. recv_counts_per_local_expert.reshape(self.ep_mesh.size(),
        # self.n_local_experts[r, l] = num tokens rank r sent to local expert l
        recv_counts_per_local_expert = funcol.all_to_all_single(
            counts, None, None, group=self.ep_mesh
        )

        # We need the list version of the counts due to NCCL sigatures. This incurs a CUDA sync.
        # TODO: avoid https://github.com/NVIDIA/nccl/issues/1648
        send_counts = (
            counts.reshape(self.ep_mesh_size, self.n_local_experts).sum(dim=1).tolist()
        )
        recv_counts = (
            recv_counts_per_local_expert.reshape(
                self.ep_mesh_size, self.n_local_experts
            )
            .sum(dim=1)
            .tolist()
        )

        # Receive toks from other workers
        x_recv = funcol.all_to_all_single_autograd(
            x_by_expert, recv_counts, send_counts, group=self.ep_mesh
        )

        # Prepare outputs
        x_send = torch.empty_like(x_recv)

        # Need to know which idxs in x_recv correspond to which local experts. Can derive from
        # recv_counts_per_local_expert.
        local_expert_idxs = (
            torch.arange(
                recv_counts_per_local_expert.numel(),
                device=recv_counts_per_local_expert.device,
            )
            % self.n_local_experts
        )
        local_expert_idxs = (
            local_expert_idxs.repeat_interleave(recv_counts_per_local_expert)
            + self.experts_start_idx
        )

        for exp_idx in range(self.experts_start_idx, self.experts_end_idx):
            idxs = local_expert_idxs == exp_idx
            # TODO: @goon - handle no-tokens edge case
            x_send[idxs] = self.experts[str(exp_idx)](x_recv[idxs])

        # Send results back to original ranks (reversed send/recv count data)
        x_out = funcol.all_to_all_single_autograd(
            x_send, send_counts, recv_counts, group=self.ep_mesh
        )

        # Store the unsorted results back in x_by_expert
        x_by_expert[flat_sorted_indices] = x_out
        # Reshape and weight
        x_by_expert = x_by_expert.reshape(*(weights.shape + x_by_expert.shape[-1:]))
        z = torch.bmm(weights[:, None], x_by_expert).squeeze(1)
        return z
