"""Rollback selects the newest numeric checkpoint across filename widths."""

from pathlib import Path

import pytest
import torch

from dama.ai.ml.trainer import Trainer, TrainingConfig, TrainingStats


@pytest.mark.parametrize(
    "steps, prune_latest",
    [
        ((999, 1000), False),
        ((999999, 1000000), False),
        ((9999999, 10000000), False),
        ((999999, 1000000, 1000001), True),
    ],
    ids=["within-padding", "seven-digits", "eight-digits", "pruned-seven-digits"],
)
def test_rollback_restores_newest_numeric_checkpoint(tmp_path, steps, prune_latest):
    holder = object.__new__(Trainer)
    holder.config = TrainingConfig(checkpoint_dir=str(tmp_path))
    holder.model = torch.nn.Linear(2, 1)
    holder.optimizer = torch.optim.AdamW(holder.model.parameters())
    holder.device = torch.device("cpu")
    holder.scheduler = holder.scaler = None
    holder.stats = TrainingStats(total_steps=max(steps) + 10)
    holder.step = max(steps) + 10
    holder.epoch = 10

    saved_states = {}
    for index, step in enumerate(steps):
        source = torch.nn.Linear(2, 1)
        with torch.no_grad():
            for parameter in source.parameters():
                parameter.fill_(index + 1)
        saved_states[step] = source.state_dict()
        torch.save(
            {"model_state_dict": saved_states[step], "step": step, "epoch": index},
            tmp_path / f"model_step_{step:06d}.pt",
        )

    attempted = []
    load_checkpoint = holder._load_checkpoint

    def load_with_possible_pruning(path):
        attempted.append(Path(path))
        if prune_latest and path == str(tmp_path / f"model_step_{max(steps):06d}.pt"):
            Path(path).unlink()
        load_checkpoint(path)

    holder._load_checkpoint = load_with_possible_pruning
    holder._rollback_after_dead_epoch("numeric-order regression")

    expected_steps = sorted(steps, reverse=True)[:2 if prune_latest else 1]
    assert attempted == [tmp_path / f"model_step_{step:06d}.pt" for step in expected_steps]
    assert holder.step == expected_steps[-1]
    for key, expected in saved_states[holder.step].items():
        torch.testing.assert_close(holder.model.state_dict()[key], expected)
