"""Replay row transport keeps exact fields and reads historical pickles."""

import copy
from dataclasses import asdict, replace
import pickle
import weakref

import pytest

from dama.ai.ml import replay


@pytest.fixture
def entry():
    return replay.ReplayEntry(
        state={"current_player": 2, "pieces_p1": [[1, 2]]},
        legal_moves=[{"path": [[1, 2], [2, 3]], "captures": []}],
        chosen_index=0,
        result=-1,
        score=0.123456789,
        sample_weight=0.987654321,
        played_index=0,
        trajectory_source="algorithm",
        was_exploration=True,
        teacher_difficulty="hard",
        opening_plies=8,
        game_id="cycle-42/game-7",
    )


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_pickle_preserves_exact_fields_and_shared_references(entry, protocol):
    sibling = replace(entry, result=1)
    restored, restored_sibling = pickle.loads(
        pickle.dumps([entry, sibling], protocol=protocol)
    )
    assert asdict(restored) == asdict(entry)
    assert asdict(restored_sibling) == asdict(sibling)
    assert restored.state is restored_sibling.state
    assert restored.legal_moves is restored_sibling.legal_moves
    assert restored.score != restored.to_dict()["score"]
    assert restored.sample_weight != restored.to_dict()["sample_weight"]


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
@pytest.mark.parametrize("include_audit_fields", [False, True])
def test_reads_legacy_dictionary_pickle(entry, protocol, include_audit_fields,
                                        monkeypatch):
    # A pre-slots dataclass uses object's ordinary dictionary pickle state.
    legacy_type = type("ReplayEntry", (), {"__module__": replay.__name__})
    legacy = legacy_type()
    values = asdict(entry)
    if not include_audit_fields:
        values = {key: values[key] for key in (
            "state", "legal_moves", "chosen_index", "result",
        )}
    legacy.__dict__.update(values)
    with monkeypatch.context() as patch:
        patch.setattr(replay, "ReplayEntry", legacy_type)
        payload = pickle.dumps(legacy, protocol=protocol)
    restored = pickle.loads(payload)
    expected = replay.ReplayEntry(**values)
    assert asdict(restored) == asdict(expected)
    assert restored.to_dict() == expected.to_dict()


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_dictionary_pickle_remains_readable_by_legacy_class(entry, protocol,
                                                          monkeypatch):
    payload = pickle.dumps(entry, protocol=protocol)
    legacy_type = type("ReplayEntry", (), {"__module__": replay.__name__})
    with monkeypatch.context() as patch:
        patch.setattr(replay, "ReplayEntry", legacy_type)
        restored = pickle.loads(payload)
    assert restored.__dict__ == asdict(entry)


def test_copy_mutability_and_weak_references(entry):
    shallow = copy.copy(entry)
    deep = copy.deepcopy(entry)
    assert asdict(shallow) == asdict(deep) == asdict(entry)
    assert shallow.state is entry.state
    assert shallow.legal_moves is entry.legal_moves
    assert deep.state is not entry.state
    assert deep.legal_moves is not entry.legal_moves
    shallow.score = 0.5
    shallow.sample_weight = 0.75
    shallow.result = 1
    assert entry.score == 0.123456789
    assert entry.sample_weight == 0.987654321
    assert entry.result == -1
    assert weakref.ref(entry)() is entry


def test_subclass_copy_retains_additional_metadata(entry):
    class AnnotatedEntry(replay.ReplayEntry):
        pass

    annotated = AnnotatedEntry(**asdict(entry))
    annotated.annotation = {"source": "test"}
    restored = copy.deepcopy(annotated)
    assert asdict(restored) == asdict(entry)
    assert restored.annotation == annotated.annotation
    assert restored.annotation is not annotated.annotation
