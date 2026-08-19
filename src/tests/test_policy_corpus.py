import json
import gzip
from pathlib import Path

import pytest

from dama.ai.ml.corpus import (
    CorpusSnapshotManager,
    analyze_replay_files,
    canonical_state_key,
    split_replay_by_file,
)


def _state(index: int, turn: int = 1) -> dict:
    row = (index // 4) % 8
    col = (index * 2 + 1 - (row % 2)) % 8
    return {
        "p1_men": [[row, col]],
        "p1_kings": [],
        "p2_men": [[7 - row, 7 - col]],
        "p2_kings": [],
        "turn": turn,
        "move_count": index,
    }


def _entry(index: int, *, state: dict | None = None, forced: bool = False) -> dict:
    moves = [{"path": [[0, 1], [1, 0]], "captures": [], "promotion": False}]
    if not forced:
        moves.append({"path": [[0, 1], [1, 2]], "captures": [], "promotion": False})
    return {
        "state": state or _state(index),
        "legal_moves": moves,
        "chosen_index": 0,
        "result": 0,
        "trajectory_source": "algorithm",
    }


def _write_replay(path: Path, entries: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries),
        encoding="utf-8",
    )


def test_canonical_state_ignores_move_count_and_normalizes_player_two() -> None:
    p1 = _state(0, turn=1)
    p1["move_count"] = 3
    p2 = {
        "p1_men": [[0, 0]],
        "p1_kings": [],
        "p2_men": [[7, 6]],
        "p2_kings": [],
        "turn": 2,
        "move_count": 99,
    }
    p1_equivalent = {
        "p1_men": [[0, 1]],
        "p1_kings": [],
        "p2_men": [[7, 7]],
        "p2_kings": [],
        "turn": 1,
        "move_count": 0,
    }

    assert canonical_state_key(p2) == canonical_state_key(p1_equivalent)
    changed_count = dict(p1_equivalent, move_count=200)
    assert canonical_state_key(changed_count) == canonical_state_key(p1_equivalent)


def test_whole_file_split_has_no_canonical_state_overlap(tmp_path: Path) -> None:
    files = []
    for index in range(8):
        path = tmp_path / f"replay_{index:02d}.jsonl"
        _write_replay(path, [_entry(index), _entry(index + 20)])
        files.append(path)

    train, validation = split_replay_by_file(files, validation_fraction=0.15, seed=77)
    train_keys = {canonical_state_key(entry.state) for entry in train}
    validation_keys = {canonical_state_key(entry.state) for entry in validation}

    assert validation
    assert train
    assert train_keys.isdisjoint(validation_keys)
    assert len(validation) % 2 == 0


def test_snapshot_gate_accepts_exact_half_fresh_and_preserves_prior(tmp_path: Path) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    snapshot_root = tmp_path / "snapshots"
    for index in range(4):
        _write_replay(replay_dir / f"replay_00_{index}.jsonl", [_entry(index, forced=index == 0)])

    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(snapshot_root),
        validation_fraction=0.25,
        split_seed=5,
        min_fresh_fraction=0.50,
    )
    settings = {"difficulty": "hard"}
    noise = {"played_action_probability": 0.10, "label_is_teacher": True}
    generation = {"algorithm_fraction": 0.70, "model_fraction": 0.30}

    first = manager.consider_snapshot(settings, noise, generation)
    assert first.admitted
    assert first.manifest_path is not None
    first_manifest_before = first.manifest_path.read_bytes()

    with first.manifest_path.open("r", encoding="utf-8") as handle:
        first_manifest = json.load(handle)
    previous_count = first_manifest["metrics"]["post_dedup_unique_state_count"]
    assert previous_count > 0

    for offset in range(previous_count):
        _write_replay(
            replay_dir / f"replay_01_{offset}.jsonl",
            [_entry(100 + offset)],
        )

    second = manager.consider_snapshot(settings, noise, generation)
    assert second.admitted
    assert second.metrics["fresh_unique_state_rate"] == pytest.approx(0.50)
    assert first.manifest_path.read_bytes() == first_manifest_before

    _write_replay(replay_dir / "replay_02_0.jsonl", [_entry(300)])
    rejected = manager.consider_snapshot(settings, noise, generation)
    assert not rejected.admitted
    assert "below" in rejected.reason
    assert rejected.manifest_path == second.manifest_path


def test_snapshot_load_removes_validation_overlap(tmp_path: Path) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    for index in range(7):
        _write_replay(replay_dir / f"replay_{index}.jsonl", [_entry(index)])

    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(tmp_path / "snapshots"),
        validation_fraction=0.15,
        split_seed=11,
        min_fresh_fraction=0.50,
    )
    external_key = canonical_state_key(_state(2))
    manager.set_external_validation_state_keys({external_key})
    decision = manager.consider_snapshot(
        {"difficulty": "hard"},
        {"played_action_probability": 0.10},
        {"algorithm_fraction": 0.70, "model_fraction": 0.30},
    )
    train, validation, manifest = manager.load_split(decision.manifest_path)

    train_keys = {canonical_state_key(entry.state) for entry in train}
    validation_keys = {canonical_state_key(entry.state) for entry in validation}
    assert train_keys.isdisjoint(validation_keys)
    assert external_key not in train_keys
    assert manifest["admission"]["passed"] is True
    assert manifest["metrics"]["forced_move_rate"] >= 0.0
    assert manifest["metrics"]["external_validation_state_count"] == 1


def test_snapshot_settings_match_rejects_previous_stage_contract(
    tmp_path: Path,
) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    for index in range(4):
        _write_replay(replay_dir / f"replay_{index}.jsonl", [_entry(index)])

    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(tmp_path / "snapshots"),
        validation_fraction=0.25,
        split_seed=17,
        min_fresh_fraction=0.50,
    )
    policy_teacher = {"stage": "policy_only", "target_type": "hard"}
    enhanced_teacher = {"stage": "enhanced", "target_type": "distribution"}
    noise = {"played_action_probability": 0.10}
    policy_generation = {"current_model_inference_depth": 1}
    enhanced_generation = {"current_model_inference_depth": 2}

    first = manager.consider_snapshot(
        policy_teacher, noise, policy_generation
    )
    rejected = manager.consider_snapshot(
        enhanced_teacher, noise, enhanced_generation
    )

    assert first.admitted
    assert not rejected.admitted
    assert rejected.manifest_path == first.manifest_path
    assert not manager.snapshot_matches_settings(
        rejected.manifest_path,
        enhanced_teacher,
        noise,
        enhanced_generation,
    )


def test_snapshot_load_rejects_tampered_training_shard(tmp_path: Path) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    for index in range(4):
        _write_replay(replay_dir / f"replay_{index}.jsonl", [_entry(index)])
    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(tmp_path / "snapshots"),
        validation_fraction=0.25,
        split_seed=19,
        min_fresh_fraction=0.50,
    )
    decision = manager.consider_snapshot({}, {}, {})
    manifest = json.loads(decision.manifest_path.read_text(encoding="utf-8"))
    shard = decision.manifest_path.parent / manifest["files"][0]["path"]
    shard.write_bytes(shard.read_bytes() + b"tampered\n")

    with pytest.raises(RuntimeError, match="integrity verification"):
        manager.load_split(decision.manifest_path)


def test_snapshot_load_rejects_tampered_state_key_digest(tmp_path: Path) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    for index in range(4):
        _write_replay(replay_dir / f"replay_{index}.jsonl", [_entry(index)])
    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(tmp_path / "snapshots"),
        validation_fraction=0.25,
        split_seed=23,
        min_fresh_fraction=0.50,
    )
    decision = manager.consider_snapshot({}, {}, {})
    manifest = json.loads(decision.manifest_path.read_text(encoding="utf-8"))
    state_keys = decision.manifest_path.parent / manifest["state_keys_file"]
    with gzip.open(state_keys, "wt", encoding="ascii", newline="\n") as handle:
        handle.write("0" * 64 + "\n")

    with pytest.raises(RuntimeError, match="canonical-state fingerprint"):
        manager.load_split(decision.manifest_path)


def test_snapshot_shards_are_independent_copies_and_report_cross_cycle_repeats(
    tmp_path: Path,
) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    state = _state(4)
    first = replay_dir / "replay_cycle_a.jsonl"
    second = replay_dir / "replay_cycle_b.jsonl"
    repeated_a = _entry(4, state=state)
    repeated_a["game_id"] = "cycle-000001-algorithm-000001"
    repeated_b = _entry(4, state=state)
    repeated_b["game_id"] = "cycle-000002-algorithm-000001"
    unique_b = _entry(200)
    unique_b["game_id"] = "cycle-000002-algorithm-000002"
    _write_replay(first, [repeated_a])
    _write_replay(second, [repeated_b, unique_b])

    metrics, _ = analyze_replay_files([first, second])
    assert metrics["cross_file_repeated_state_count"] == 1
    assert metrics["cross_file_unique_state_count"] == 1
    assert metrics["cross_cycle_repeated_state_count"] == 1
    assert metrics["cross_cycle_unique_state_count"] == 1

    manager = CorpusSnapshotManager(
        str(replay_dir), str(tmp_path / "snapshots"),
        validation_fraction=0.1, split_seed=7, min_fresh_fraction=0.0,
    )
    decision = manager.consider_snapshot({}, {}, {})
    manifest = json.loads(decision.manifest_path.read_text(encoding="utf-8"))
    shard = decision.manifest_path.parent / manifest["files"][0]["path"]
    assert manifest["files"][0]["storage"] == "copy"
    assert shard.stat().st_ino != first.stat().st_ino


def test_existing_validation_manifest_must_match_split_contract(
    tmp_path: Path,
) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    for index in range(4):
        _write_replay(replay_dir / f"replay_{index}.jsonl", [_entry(index)])
    snapshot_root = tmp_path / "snapshots"
    manager = CorpusSnapshotManager(
        str(replay_dir), str(snapshot_root), validation_fraction=0.25,
        split_seed=11, min_fresh_fraction=0.50,
    )
    manager.consider_snapshot({}, {}, {})
    manifest_path = snapshot_root / "validation" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["split"]["seed"] = 12
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="split fraction and seed"):
        manager.consider_snapshot({}, {}, {})


def _contract_entry(index: int, source: str, game_id: str) -> dict:
    entry = _entry(index)
    entry.update({
        "played_index": 0,
        "trajectory_source": source,
        "was_exploration": False,
        "teacher_difficulty": "hard",
        "opening_plies": 2,
        "game_id": game_id,
    })
    return entry


def test_recovery_snapshots_exclude_legacy_files_and_audit_exact_game_mix(
    tmp_path: Path,
) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    _write_replay(replay_dir / "replay_legacy.jsonl", [_entry(900)])
    for file_index in range(2):
        entries = []
        for game_index in range(10):
            source = "algorithm" if game_index < 7 else "current_model"
            entries.append(_contract_entry(
                file_index * 100 + game_index,
                source,
                f"{file_index}-{game_index}",
            ))
        _write_replay(replay_dir / f"replay_repaired_{file_index}.jsonl", entries)

    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(tmp_path / "snapshots"),
        validation_fraction=0.50,
        split_seed=3,
        min_fresh_fraction=0.50,
        enforce_policy_contract=True,
        allowed_opening_plies=(2, 4, 6, 8),
    )
    decision = manager.consider_snapshot(
        {"difficulty": "hard"},
        {"played_action_probability": 0.10},
        {"algorithm_fraction": 0.70, "model_fraction": 0.30},
    )
    _, _, manifest = manager.load_split(decision.manifest_path)

    assert decision.admitted
    assert "replay_legacy.jsonl" in manifest["metrics"]["rejected_replay_files"]
    assert manifest["metrics"]["source_game_counts"] == {
        "algorithm": 7,
        "current_model": 3,
    }
