import pytest
import torch
import torch.nn.functional as F

from mamba_ssm.distributed.tensor_parallel import ParallelEmbeddings


@pytest.mark.parametrize("max_position_embeddings", [0, 8])
@pytest.mark.parametrize("combine_batch_seqlen_dim", [False, True])
@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_parallel_embeddings_without_process_group(
    max_position_embeddings, combine_batch_seqlen_dim, sequence_parallel
):
    embeddings = ParallelEmbeddings(
        6, 10, max_position_embeddings, process_group=None,
        sequence_parallel=sequence_parallel,
    )
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    actual = embeddings(input_ids, combine_batch_seqlen_dim=combine_batch_seqlen_dim)
    expected = F.embedding(input_ids, embeddings.word_embeddings.weight)
    if max_position_embeddings > 0:
        expected = expected + F.embedding(torch.arange(3), embeddings.position_embeddings.weight)
    if combine_batch_seqlen_dim:
        expected = expected.flatten(0, 1)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert embeddings.word_embeddings.weight.grad is not None
    if max_position_embeddings > 0:
        assert embeddings.position_embeddings.weight.grad is not None


def test_parallel_embeddings_explicit_positions_without_process_group():
    embeddings = ParallelEmbeddings(6, 10, 8, process_group=None)
    input_ids = torch.tensor([[1, 2], [3, 4]])
    position_ids = torch.tensor([[2, 3], [4, 5]])
    expected = embeddings.word_embeddings(input_ids) + embeddings.position_embeddings(position_ids)
    torch.testing.assert_close(embeddings(input_ids, position_ids), expected)
