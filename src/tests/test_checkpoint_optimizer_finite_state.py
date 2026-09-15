"""A finite checkpoint model must not inherit unusable Adam state."""

import pytest
import torch

from dama.ai.ml.trainer import Trainer, TrainingConfig, TrainingStats


def _checkpoint_holder(tmp_path, recovery=False):
    holder = object.__new__(Trainer)
    holder.config = TrainingConfig(
        learning_rate=0.01, weight_decay=0.02, recovery_enforced=recovery)
    holder.model = torch.nn.Linear(2, 1)
    holder.device = torch.device("cpu")
    holder.optimizer = torch.optim.AdamW(
        holder.model.parameters(), lr=0.01, weight_decay=0.02)
    holder.scheduler = holder.scaler = None
    holder.stats = TrainingStats()
    inputs = torch.tensor([[0.2, 0.7], [0.9, -0.3]])
    holder.model(inputs).square().mean().backward()
    holder.optimizer.step()
    checkpoint = {
        "model_state_dict": holder.model.state_dict(),
        "optimizer_state_dict": holder.optimizer.state_dict(),
        "step": 7,
    }
    return holder, checkpoint, inputs, tmp_path / "model_step_000007.pt"


@pytest.mark.parametrize("field", ["step", "exp_avg", "exp_avg_sq"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_checkpoint_discards_nonfinite_optimizer_state(
    tmp_path, field, invalid, capsys,
):
    holder, checkpoint, inputs, path = _checkpoint_holder(tmp_path, recovery=True)
    saved_state = next(iter(checkpoint["optimizer_state_dict"]["state"].values()))
    saved_state[field].fill_(invalid)
    torch.save(checkpoint, path)
    reference = torch.nn.Linear(2, 1)
    reference.load_state_dict(checkpoint["model_state_dict"])
    fresh_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01, weight_decay=0.02)

    holder._load_checkpoint(str(path))

    # Exercise the next real update, not just the loader's return value.
    for model, optimizer in ((holder.model, holder.optimizer),
                             (reference, fresh_optimizer)):
        optimizer.zero_grad(set_to_none=True)
        model(inputs).square().mean().backward()
        optimizer.step()
    for actual, expected in zip(holder.model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected)
    assert holder.step == 7
    message = capsys.readouterr().out
    assert "non-finite" in message.lower()
    assert field in message
    assert "fresh optimizer" in message


@pytest.mark.parametrize("recovery", [False, True])
def test_finite_optimizer_state_keeps_exact_next_update(tmp_path, recovery):
    holder, checkpoint, inputs, path = _checkpoint_holder(tmp_path, recovery)
    torch.save(checkpoint, path)
    reference = torch.nn.Linear(2, 1)
    reference.load_state_dict(checkpoint["model_state_dict"])
    reference_optimizer = torch.optim.AdamW(reference.parameters())
    reference_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    holder._load_checkpoint(str(path))

    for model, optimizer in ((holder.model, holder.optimizer),
                             (reference, reference_optimizer)):
        optimizer.zero_grad(set_to_none=True)
        model(inputs).square().mean().backward()
        optimizer.step()
    for actual, expected in zip(holder.model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected)
