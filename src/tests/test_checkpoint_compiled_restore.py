"""Dead-epoch rollback restores weights through a compiled model wrapper."""

import pytest
import torch

from dama.ai.ml.trainer import Trainer, TrainingConfig, TrainingStats


@pytest.mark.parametrize("saved_compiled", [False, True])
@pytest.mark.parametrize("live_compiled", [False, True])
def test_rollback_restores_weights_and_optimizer_for_compiled_models(
    tmp_path, saved_compiled, live_compiled,
):
    config = TrainingConfig(
        checkpoint_dir=str(tmp_path), learning_rate=0.01, weight_decay=0.02)
    source = torch.nn.Linear(2, 1)
    source_optimizer = torch.optim.AdamW(
        source.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay)
    inputs = torch.tensor([[0.2, 0.7], [0.9, -0.3]])
    source(inputs).square().mean().backward()
    source_optimizer.step()
    saved_model = torch.compile(source, backend="eager") if saved_compiled else source
    checkpoint = tmp_path / "model_step_000001.pt"
    torch.save({
        "model_state_dict": saved_model.state_dict(),
        "optimizer_state_dict": source_optimizer.state_dict(),
        "step": 1, "epoch": 1,
    }, checkpoint)

    live = torch.nn.Linear(2, 1)
    with torch.no_grad():
        for parameter in live.parameters():
            parameter.fill_(10)
    holder = object.__new__(Trainer)
    holder.config = config
    holder.model = torch.compile(live, backend="eager") if live_compiled else live
    original_wrapper = holder.model
    holder.optimizer = torch.optim.AdamW(
        holder.model.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay)
    holder.device = torch.device("cpu")
    holder.scheduler = holder.scaler = None
    holder.stats = TrainingStats(total_steps=10, epochs_completed=10)
    holder.step = holder.epoch = 10

    holder._rollback_after_dead_epoch("test")

    assert holder.model is original_wrapper
    assert holder.step == holder.epoch == 1
    for actual, expected in zip(live.parameters(), source.parameters()):
        torch.testing.assert_close(actual, expected)
        for key, value in source_optimizer.state[expected].items():
            torch.testing.assert_close(holder.optimizer.state[actual][key], value)

    # Restored parameters still belong to the live optimizer and wrapper.
    for model, optimizer in ((holder.model, holder.optimizer),
                             (source, source_optimizer)):
        optimizer.zero_grad(set_to_none=True)
        model(inputs).square().mean().backward()
        optimizer.step()
    for actual, expected in zip(live.parameters(), source.parameters()):
        torch.testing.assert_close(actual, expected)
