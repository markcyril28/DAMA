"""Legacy checkpoint shapes must preserve supported model architectures."""

import pytest
import torch

from dama.ai.ml.model import create_model, load_model


@pytest.mark.parametrize("num_blocks", [0, 1])
@pytest.mark.parametrize("checkpoint_format", ["raw", "wrapped", "compiled"])
def test_legacy_checkpoint_preserves_residual_block_count(
    tmp_path, num_blocks, checkpoint_format
):
    """A backbone without residual blocks is valid, including without metadata."""
    original = create_model(
        channels=4,
        num_blocks=num_blocks,
        embedding_size=8,
        hidden_size=6,
        value_head_enabled=True,
        value_head_hidden=5,
    ).eval()
    state = original.state_dict()
    if checkpoint_format == "compiled":
        state = {"_orig_mod." + name: value for name, value in state.items()}
    checkpoint = state if checkpoint_format == "raw" else {"model_state_dict": state}
    path = tmp_path / "legacy.pt"
    torch.save(checkpoint, path)

    with pytest.warns(RuntimeWarning, match="encoding version 1"):
        restored = load_model(str(path), torch.device("cpu"))

    assert restored.arch_params == original.arch_params
    assert len(restored.board_encoder.blocks) == num_blocks
    assert not restored.training
    for name, expected in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], expected, rtol=0, atol=0)
