"""Checkpoint optimizer failures must discard live momentum before rollback."""

import pytest
import torch

from dama.ai.ml.trainer import Trainer, TrainingConfig, TrainingStats


@pytest.mark.parametrize("saved_state", ["missing", "incompatible"])
@pytest.mark.parametrize("invalid_momentum", [False, True])
def test_checkpoint_optimizer_fallback_clears_previous_training_state(
    tmp_path, saved_state, invalid_momentum,
):
    holder = object.__new__(Trainer)
    holder.config = TrainingConfig(learning_rate=0.01, weight_decay=0.02)
    holder.model = torch.nn.Linear(2, 1)
    holder.device = torch.device("cpu")
    holder.optimizer = torch.optim.AdamW(
        holder.model.parameters(), lr=holder.config.learning_rate,
        weight_decay=holder.config.weight_decay)
    holder.scheduler = holder.scaler = None
    holder.stats = TrainingStats()
    inputs = torch.tensor([[0.2, 0.7], [0.9, -0.3]])
    holder.model(inputs).square().mean().backward()
    holder.optimizer.step()
    assert holder.optimizer.state
    if invalid_momentum:
        for state in holder.optimizer.state.values():
            state["exp_avg"].fill_(float("nan"))

    reference = torch.nn.Linear(2, 1)
    checkpoint = {"model_state_dict": reference.state_dict(), "step": 1}
    if saved_state == "incompatible":
        # A previous architecture used a different parameter-group layout.
        checkpoint["optimizer_state_dict"] = {"state": {}, "param_groups": []}
    path = tmp_path / "model_step_000001.pt"
    torch.save(checkpoint, path)

    holder._load_checkpoint(str(path))

    assert not holder.optimizer.state
    reference_optimizer = torch.optim.AdamW(
        reference.parameters(), lr=holder.config.learning_rate,
        weight_decay=holder.config.weight_decay)
    for model, optimizer in ((holder.model, holder.optimizer),
                             (reference, reference_optimizer)):
        optimizer.zero_grad(set_to_none=True)
        model(inputs).square().mean().backward()
        optimizer.step()
    for actual, expected in zip(holder.model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected)
