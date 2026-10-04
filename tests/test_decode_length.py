from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from mamba_ssm.utils import generation
from mamba_ssm.utils.generation import decode


@pytest.mark.parametrize("max_length", [2, 3])
@pytest.mark.parametrize("cg", [False, True])
def test_decode_does_not_extend_prompt_at_length_limit(max_length, cg, monkeypatch):
    monkeypatch.setattr(torch.cuda, "Event", Mock())
    monkeypatch.setattr(generation, "update_graph_cache",
                        Mock(side_effect=AssertionError("Cache should not be allocated")))
    input_ids = torch.tensor([[1, 2, 3], [3, 2, 1]])
    model = Mock(side_effect=AssertionError("Model should not run"))
    streamer = Mock()
    output = decode(input_ids, model, max_length, cg=cg, streamer=streamer,
                    output_scores=True)
    torch.testing.assert_close(output.sequences, input_ids)
    assert output.scores == ()
    model.assert_not_called()
    streamer.put.assert_called_once()
    torch.testing.assert_close(streamer.put.call_args.args[0], input_ids)
    streamer.end.assert_called_once_with()


def test_decode_generates_up_to_length_limit(monkeypatch):
    # Timing events are unrelated to this CPU-only decoding test.
    monkeypatch.setattr(torch.cuda, "Event", Mock())
    model = Mock(return_value=SimpleNamespace(logits=torch.tensor([[[0., 1., 0.]]])))
    output = decode(torch.tensor([[2, 0]]), model, max_length=4, output_scores=True)
    torch.testing.assert_close(output.sequences, torch.tensor([[2, 0, 1, 1]]))
    assert len(output.scores) == 2
    assert model.call_count == 2
