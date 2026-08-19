import pytest
import torch

from dama.ai.ml.replay import ReplayEntry
from dama.ai.ml.teacher_targets import (
    build_teacher_target,
    soft_target_cross_entropy,
    soft_teacher_policy,
)
from dama.game_state import GameState


def test_soft_teacher_policy_is_normalized_and_blends_hard_label() -> None:
    unblended = soft_teacher_policy((2.0, 1.0, 0.0), hard_label_blend=0.0)
    blended = soft_teacher_policy(
        (2.0, 1.0, 0.0), hard_index=2, hard_label_blend=0.25)

    assert sum(blended) == pytest.approx(1.0)
    assert blended[2] > unblended[2]
    assert 0.0 < blended[0] < 1.0


def test_soft_target_cross_entropy_rewards_matching_distribution() -> None:
    teacher = torch.tensor([[0.8, 0.2, 0.0]], dtype=torch.float32)
    counts = torch.tensor([2])
    matching = soft_target_cross_entropy(
        torch.log(torch.tensor([[0.8, 0.2, 1.0]])), counts, teacher)
    reversed_loss = soft_target_cross_entropy(
        torch.log(torch.tensor([[0.2, 0.8, 1.0]])), counts, teacher)

    assert matching < reversed_loss


def test_teacher_target_contains_every_legal_move_and_bounded_value() -> None:
    state = GameState.initial()
    entry = ReplayEntry(
        state=state.to_compact(),
        legal_moves=[move.to_dict() for move in state.legal_moves()],
        chosen_index=0,
        result=0,
    )
    target = build_teacher_target(
        entry,
        depth=1,
        temperature=1.0,
        value_scale=1000.0,
        hard_label_blend=0.25,
    )

    assert len(target.move_values) == len(entry.legal_moves)
    assert len(target.policy) == len(entry.legal_moves)
    assert sum(target.policy) == pytest.approx(1.0)
    assert -1.0 <= target.state_value <= 1.0
