# Copyright (c) 2024, Tri Dao.
# The TensorParallel linear modules are inspired by https://github.com/NVIDIA/apex/blob/master/apex/transformer/tensor_parallel/layers.py
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.distributed import ProcessGroup

from einops import rearrange

from mamba_ssm.distributed.distributed_utils import (
    all_gather_raw,
    all_reduce,
    all_reduce_raw,
    reduce_scatter,
    reduce_scatter_raw,
)


class ParallelLinearFunc(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, x, weight, bias, process_group=None, sequence_parallel=True):
        """
        If process_group is not None and sequence_parallel=True, we're doing Tensor Parallel
        with sequence parallelism: we do an all_gather_raw of x before doing the matmul.
        """
        ctx.compute_weight_gradient = weight.requires_grad
        ctx.process_group = process_group
        ctx.sequence_parallel = sequence_parallel

        if torch.is_autocast_enabled():
            x = x.to(dtype=torch.get_autocast_gpu_dtype())
        x = x.contiguous()
        if process_group is not None and sequence_parallel:
            # We want to kick off the all_gather early, before weight dtype conversion
            total_x, handle_x = all_gather_raw(x, process_group, async_op=True)
        else:
            total_x = x

        if torch.is_autocast_enabled():
            weight = weight.to(dtype=torch.get_autocast_gpu_dtype())
            bias = bias.to(dtype=torch.get_autocast_gpu_dtype()) if bias is not None else None
        weight = weight.contiguous()
        if process_group is not None and sequence_parallel:
            handle_x.wait()
        batch_shape, n = total_x.shape[:-1], total_x.shape[-1]
        batch_dim = batch_shape.numel()
        # https://github.com/pytorch/pytorch/blob/5b51849b48a7dbccd297286cc0110def4706f9e7/aten/src/ATen/native/cuda/Blas.cpp#L174
        output = F.linear(total_x, weight, bias)
        if ctx.compute_weight_gradient:
            ctx.save_for_backward(x, weight)
        else:
            ctx.save_for_backward(weight)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        process_group = ctx.process_group
        sequence_parallel = ctx.sequence_parallel
        if ctx.compute_weight_gradient:
            x, weight = ctx.saved_tensors
            if process_group is not None and sequence_parallel:
                total_x, handle_x = all_gather_raw(x, process_group, async_op=True)
            else:
                total_x = x
        else:
            (weight,) = ctx.saved_tensors
            total_x = None
        batch_shape = grad_output.shape[:-1]
        batch_dim = batch_shape.numel()
        grad_output = grad_output.reshape(batch_dim, grad_output.shape[-1])
        if ctx.needs_input_grad[0]:
            grad_input = F.linear(grad_output, weight.t())
            grad_input = grad_input.reshape(*batch_shape, grad_input.shape[-1])
            if process_group is not None:
                reduce_fn = reduce_scatter_raw if sequence_parallel else all_reduce_raw
                grad_input, handle_grad_input = reduce_fn(grad_input, process_group, async_op=True)
        else:
            grad_input = None
        if ctx.needs_input_grad[1]:
            assert ctx.compute_weight_gradient
            if process_group is not None and sequence_parallel:
                handle_x.wait()
            grad_weight = torch.einsum(
                "bo,bi->oi", grad_output, total_x.reshape(batch_dim, total_x.shape[-1])
            )
        else:
            grad_weight = None
        grad_bias = grad_output.sum(dim=0) if ctx.needs_input_grad[2] else None
        if process_group is not None and ctx.needs_input_grad[0]:
            handle_grad_input.wait()
        return grad_input, grad_weight, grad_bias, None, None


def parallel_linear_func(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    process_group: Optional[ProcessGroup] = None,
    sequence_parallel: bool = True,
):
    return ParallelLinearFunc.apply(x, weight, bias, process_group, sequence_parallel)


class ColumnParallelLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        process_group: ProcessGroup,
        bias: bool = True,
        sequence_parallel=True,
        multiple_of=1,
        device=None,
        dtype=None,
    ) -> None:
        world_size = torch.distributed.get_world_size(process_group)
        if out_features % multiple_of:
            raise ValueError(f"out_features ({out_features}) must be a multiple of {multiple_of}")
        multiple = out_features // multiple_of
        # We want to split @multiple across world_size, but it could be an uneven split
        div = multiple // world_size
        mod = multiple % world_size
        # The first @mod ranks get @div + 1 copies, the rest get @div copies
        local_multiple = div + int(torch.distributed.get_rank(process_group) < mod)
        super().__init__(
            in_features, local_multiple * multiple_of, bias=bias, device=device, dtype=dtype
        )
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel

    def forward(self, x):
        # If self.sequence_parallel is True, we're doing Tensor Parallel with sequence parallelism:
        # we do an all_gather of x before doing the matmul.
        # If not, then the input is already gathered.
        return parallel_linear_func(
            x,
            self.weight,
            self.bias,
            process_group=self.process_group,
            sequence_parallel=self.sequence_parallel,
        )


class RowParallelLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        process_group: ProcessGroup,
        bias: bool = True,
        sequence_parallel=True,
        multiple_of=1,
        device=None,
        dtype=None,
    ) -> None:
        world_size = torch.distributed.get_world_size(process_group)
        rank = torch.distributed.get_rank(process_group)
        if in_features % multiple_of:
            raise ValueError(f"in_features ({in_features}) must be a multiple of {multiple_of}")
        multiple = in_features // multiple_of
        # We want to split @multiple across world_size, but it could be an uneven split
        div = multiple // world_size
        mod = multiple % world_size
        # The first @mod ranks get @div + 1 copies, the rest get @div copies
        local_multiple = div + int(torch.distributed.get_rank(process_group) < mod)
        # Only rank 0 will have bias
        super().__init__(
            local_multiple * multiple_of,
            out_features,
            bias=bias and rank == 0,
            device=device,
            dtype=dtype,
        )
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel

    def forward(self, x):
        """
        We're doing Tensor Parallel with sequence parallelism: we do the matmul and then
        a reduce_scatter of the result.
        """
        out = parallel_linear_func(x, self.weight, self.bias)
        reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
        return reduce_fn(out, self.process_group)


class VocabParallelEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, *args, process_group=None, padding_idx=None, **kwargs):
        self.process_group = process_group
        if process_group is not None:
            world_size = torch.distributed.get_world_size(process_group)
            if num_embeddings % world_size != 0:
                raise ValueError(
                    f"num_embeddings ({num_embeddings}) must be divisible by "
                    f"world_size ({world_size})"
                )
            if world_size > 1 and padding_idx is not None:
                raise RuntimeError("ParallelEmbedding does not support padding_idx")
        else:
            world_size = 1
        super().__init__(num_embeddings // world_size, *args, padding_idx=padding_idx, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        if self.process_group is None:
            return super().forward(input)
        else:
            rank = torch.distributed.get_rank(self.process_group)
            vocab_size = self.num_embeddings
            vocab_start_index, vocab_end_index = rank * vocab_size, (rank + 1) * vocab_size
            # Create a mask of valid vocab ids (1 means it needs to be masked).
            input_ids_mask = (input < vocab_start_index) | (input >= vocab_end_index)
            input = input - vocab_start_index
            input[input_ids_mask] = 0
            embeddings = super().forward(input)
            embeddings[input_ids_mask] = 0.0
            return embeddings


class ColumnParallelEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, *args, process_group=None, **kwargs):
        self.process_group = process_group
        if process_group is not None:
            world_size = torch.distributed.get_world_size(process_group)
            if embedding_dim % world_size != 0:
                raise ValueError(
                    f"embedding_dim ({embedding_dim}) must be divisible by "
                    f"world_size ({world_size})"
                )
        else:
            world_size = 1
        super().__init__(num_embeddings, embedding_dim // world_size, *args, **kwargs)


class ParallelEmbeddings(nn.Module):
    def __init__(
        self,
        embed_dim,
        vocab_size,
        max_position_embeddings,
        process_group,
        padding_idx=None,
        sequence_parallel=True,
        device=None,
        dtype=None,
    ):
        """
        If max_position_embeddings <= 0, there's no position embeddings
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.word_embeddings = VocabParallelEmbedding(
            vocab_size,
            embed_dim,
            padding_idx=padding_idx,
            process_group=process_group,
            **factory_kwargs,
        )
        self.max_position_embeddings = max_position_embeddings
        if self.max_position_embeddings > 0:
            self.position_embeddings = ColumnParallelEmbedding(
                max_position_embeddings, embed_dim, process_group=process_group, **factory_kwargs
            )

    def forward(self, input_ids, position_ids=None, combine_batch_seqlen_dim=False):
        """
        input_ids: (batch, seqlen)
        position_ids: (batch, seqlen)
        """
        batch_size, seqlen = input_ids.shape
        world_size = torch.distributed.get_world_size(self.process_group)
        embeddings = self.word_embeddings(input_ids)
        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = torch.arange(seqlen, dtype=torch.long, device=input_ids.device)
            position_embeddings = self.position_embeddings(position_ids)
            if world_size <= 1:
                embeddings = embeddings + position_embeddings
            else:
                partition_dim = self.position_embeddings.embedding_dim
                rank = torch.distributed.get_rank(self.process_group)
                embeddings[
                    ..., rank * partition_dim : (rank + 1) * partition_dim
                ] += position_embeddings
        if combine_batch_seqlen_dim:
            embeddings = rearrange(embeddings, "b s d -> (b s) d")
        reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
        return embeddings if world_size <= 1 else reduce_fn(embeddings, self.process_group)
