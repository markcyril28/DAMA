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
        # This case asserts the exact freshness arithmetic of the admission
        # gate, so the hold-out is pinned; growth has its own coverage.
        grow_holdout=False,
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


def _admit_series(
    manager: CorpusSnapshotManager,
    replay_dir: Path,
    cycles: int,
    start_cycle: int = 0,
) -> list:
    """Admit ``cycles`` snapshots, feeding all-fresh states each time."""
    settings = {"difficulty": "hard"}
    noise = {"played_action_probability": 0.10, "label_is_teacher": True}
    generation = {"algorithm_fraction": 0.70, "model_fraction": 0.30}
    decisions = []
    for cycle in range(start_cycle, start_cycle + cycles):
        # The live pipeline rotates shards out via cleanup_old_files, so each
        # cycle presents an all-fresh corpus and clears the freshness gate.
        for stale in replay_dir.glob("replay_*.jsonl"):
            stale.unlink()
        for offset in range(4):
            _write_replay(
                replay_dir / f"replay_{cycle:02d}_{offset}.jsonl",
                [_entry(1000 * (cycle + 1) + offset)],
            )
        decision = manager.consider_snapshot(settings, noise, generation)
        assert decision.admitted, decision.reason
        decisions.append(decision)
    return decisions


def test_snapshot_retention_prunes_oldest_and_keeps_current(tmp_path: Path) -> None:
    """A positive retention cap must reclaim disk without breaking the active split.

    Each admission stores a full copy of the replay corpus, so an uncapped
    snapshot root grows without bound and fills the volume mid-run.
    """
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    snapshot_root = tmp_path / "snapshots"
    for index in range(4):
        _write_replay(replay_dir / f"replay_seed_{index}.jsonl", [_entry(index)])

    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(snapshot_root),
        validation_fraction=0.25,
        split_seed=5,
        min_fresh_fraction=0.50,
        max_retained_snapshots=3,
    )
    decisions = _admit_series(manager, replay_dir, 5)

    kept = sorted(path.name for path in snapshot_root.glob("snapshot_v*"))
    assert kept == ["snapshot_v000003", "snapshot_v000004", "snapshot_v000005"]

    # The newest admission stays intact and remains loadable.
    current = decisions[-1]
    assert current.manifest_path is not None
    assert current.manifest_path.is_file()
    train_entries, _validation_entries, _manifest = manager.load_split()
    assert train_entries

    # Version numbering must not restart after pruning.
    assert manager._next_version() == 6
    later = _admit_series(manager, replay_dir, 1, start_cycle=5)[0]
    assert later.manifest_path is not None
    assert later.manifest_path.parent.name == "snapshot_v000006"

    # The frozen validation set is never a pruning candidate.
    assert (snapshot_root / "validation" / "manifest.json").is_file()


def test_snapshot_retention_disabled_by_default_keeps_every_snapshot(
    tmp_path: Path,
) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    snapshot_root = tmp_path / "snapshots"
    for index in range(4):
        _write_replay(replay_dir / f"replay_seed_{index}.jsonl", [_entry(index)])

    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(snapshot_root),
        validation_fraction=0.25,
        split_seed=5,
        min_fresh_fraction=0.50,
    )
    assert manager.max_retained_snapshots == 0
    _admit_series(manager, replay_dir, 4)
    assert len(list(snapshot_root.glob("snapshot_v*"))) == 4


def test_snapshot_retention_rejects_negative_cap(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        CorpusSnapshotManager(
            str(tmp_path / "replay"),
            str(tmp_path / "snapshots"),
            max_retained_snapshots=-1,
        )


def _validation_manifest(snapshot_root: Path) -> dict:
    return json.loads(
        (snapshot_root / "validation" / "manifest.json").read_text(encoding="utf-8")
    )


def test_validation_holdout_grows_append_only_toward_configured_share(
    tmp_path: Path,
) -> None:
    """Audit finding F3: a frozen hold-out decays far below its approved share.

    The split is created from whatever files exist at creation time and, before
    this behaviour, never grew -- so an approved 15% realized 1.7% against a
    rolling corpus while the manifest still declared 0.15.
    """
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    snapshot_root = tmp_path / "snapshots"
    for index in range(2):
        _write_replay(replay_dir / f"replay_seed_{index}.jsonl", [_entry(index)])

    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(snapshot_root),
        validation_fraction=0.25,
        split_seed=5,
        min_fresh_fraction=0.0,
    )
    _admit_series(manager, replay_dir, 1)
    created = _validation_manifest(snapshot_root)
    assert len(created["files"]) == 1
    held_at_creation = {record["name"] for record in created["files"]}

    # Grow the corpus well beyond the creation-time size.
    for index in range(18):
        _write_replay(replay_dir / f"replay_grow_{index:02d}.jsonl", [_entry(500 + index)])
    files = sorted(replay_dir.glob("replay_*.jsonl"))
    manifest, _keys = manager._ensure_validation(files)

    split = manifest["split"]
    assert split["fraction"] == 0.25
    assert split["validation_file_count"] == round(len(files) * 0.25)
    assert split["realized_file_fraction"] == pytest.approx(0.25, abs=0.03)

    # Append-only: every originally held file is still held.
    held_now = {record["name"] for record in manifest["files"]}
    assert held_at_creation <= held_now
    assert manifest["growth_history"][-1]["source_file_count"] == len(files)

    # The regenerated key set must satisfy the integrity contract.
    reloaded, _state_keys = manager._ensure_validation(files)
    assert len(reloaded["files"]) == len(manifest["files"])


def test_validation_growth_never_releases_a_held_file(tmp_path: Path) -> None:
    """States may move train -> validation, never the reverse."""
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    snapshot_root = tmp_path / "snapshots"
    for index in range(2):
        _write_replay(replay_dir / f"replay_seed_{index}.jsonl", [_entry(index)])

    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(snapshot_root),
        validation_fraction=0.30,
        split_seed=11,
        min_fresh_fraction=0.0,
    )
    _admit_series(manager, replay_dir, 1)
    held = {record["name"] for record in _validation_manifest(snapshot_root)["files"]}

    for round_index in range(4):
        for index in range(5):
            _write_replay(
                replay_dir / f"replay_r{round_index}_{index}.jsonl",
                [_entry(2000 + round_index * 100 + index)],
            )
        files = sorted(replay_dir.glob("replay_*.jsonl"))
        manifest, _keys = manager._ensure_validation(files)
        now = {record["name"] for record in manifest["files"]}
        assert held <= now, "growth released a previously held validation file"
        held = now
        # A training file must always survive the split.
        assert len(now) < len(files)


def test_validation_holdout_growth_can_be_disabled(tmp_path: Path) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    snapshot_root = tmp_path / "snapshots"
    for index in range(2):
        _write_replay(replay_dir / f"replay_seed_{index}.jsonl", [_entry(index)])

    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(snapshot_root),
        validation_fraction=0.25,
        split_seed=5,
        min_fresh_fraction=0.0,
        grow_holdout=False,
    )
    _admit_series(manager, replay_dir, 1)
    for index in range(18):
        _write_replay(replay_dir / f"replay_grow_{index:02d}.jsonl", [_entry(500 + index)])
    files = sorted(replay_dir.glob("replay_*.jsonl"))
    manifest, _keys = manager._ensure_validation(files)

    assert len(manifest["files"]) == 1
    assert "growth_history" not in manifest
    # The realized share is still reported honestly.
    assert manifest["split"]["realized_file_fraction"] == pytest.approx(1 / len(files))


def test_windows_written_relative_paths_still_resolve(tmp_path: Path) -> None:
    """Finding F5: a backslash-separated pointer must not reset the lineage.

    The native-Windows launcher wrote `..\\validation\\manifest.json` into
    snapshot v11. On WSL that path does not resolve, current_manifest_path()
    returns None, and consider_snapshot takes the no-previous-corpus branch --
    skipping the >=50% freshness floor, which is how v12 was admitted.
    """
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    snapshot_root = tmp_path / "snapshots"
    for index in range(4):
        _write_replay(replay_dir / f"replay_seed_{index}.jsonl", [_entry(index)])
    manager = CorpusSnapshotManager(
        str(replay_dir), str(snapshot_root),
        validation_fraction=0.25, split_seed=5,
        min_fresh_fraction=0.50, grow_holdout=False,
    )
    settings = {"difficulty": "hard"}
    noise = {"played_action_probability": 0.10, "label_is_teacher": True}
    generation = {"algorithm_fraction": 0.70, "model_fraction": 0.30}
    first = manager.consider_snapshot(settings, noise, generation)
    assert first.admitted

    # Everything written must be POSIX-separated.
    pointer = (snapshot_root / "CURRENT").read_text(encoding="utf-8").strip()
    assert "\\" not in pointer
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert "\\" not in manifest["validation_manifest"]

    # A pointer left behind by Windows must still resolve on this host.
    (snapshot_root / "CURRENT").write_text(
        pointer.replace("/", "\\") + "\n", encoding="utf-8")
    assert manager.current_manifest_path() is not None, (
        "backslash pointer did not resolve -- the lineage would silently reset")

    # And the freshness gate must therefore still be enforced.
    _write_replay(replay_dir / "replay_new.jsonl", [_entry(950)])
    decision = manager.consider_snapshot(settings, noise, generation)
    assert not decision.admitted
    assert "below" in decision.reason


def test_lost_current_pointer_fails_closed_instead_of_skipping_freshness_gate(
    tmp_path: Path,
) -> None:
    """A vanished CURRENT pointer must not silently disable the freshness gate.

    The gate reads ``current_path is not None``, so before this a lost pointer
    made every candidate admissible and recorded a 100% fresh rate that is only
    an artifact of an empty previous-key set -- which is how snapshot_v000012
    entered the policy-distillation namespace ungated.
    """
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    snapshot_root = tmp_path / "snapshots"
    for index in range(4):
        _write_replay(replay_dir / f"replay_seed_{index}.jsonl", [_entry(index)])

    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(snapshot_root),
        validation_fraction=0.25,
        split_seed=5,
        min_fresh_fraction=0.50,
        grow_holdout=False,
    )
    settings = {"difficulty": "hard"}
    noise = {"played_action_probability": 0.10, "label_is_teacher": True}
    generation = {"algorithm_fraction": 0.70, "model_fraction": 0.30}

    first = manager.consider_snapshot(settings, noise, generation)
    assert first.admitted

    # Simulate the pointer loss observed in production.
    (snapshot_root / "CURRENT").unlink()
    (snapshot_root / "current.json").unlink()

    _write_replay(replay_dir / "replay_new.jsonl", [_entry(900)])
    with pytest.raises(RuntimeError, match="pointer is missing"):
        manager.consider_snapshot(settings, noise, generation)

    # Restoring the pointer restores normal gated behaviour.
    (snapshot_root / "CURRENT").write_text(
        "snapshot_v000001/manifest.json\n", encoding="utf-8")
    decision = manager.consider_snapshot(settings, noise, generation)
    assert not decision.admitted
    assert "below" in decision.reason


def test_first_ever_admission_still_allowed_without_a_pointer(
    tmp_path: Path,
) -> None:
    """The fail-closed check must not break a genuinely empty namespace."""
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    snapshot_root = tmp_path / "snapshots"
    for index in range(4):
        _write_replay(replay_dir / f"replay_seed_{index}.jsonl", [_entry(index)])

    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(snapshot_root),
        validation_fraction=0.25,
        split_seed=5,
        min_fresh_fraction=0.50,
        grow_holdout=False,
    )
    assert manager.current_manifest_path() is None
    assert manager._snapshot_dirs() == []
    decision = manager.consider_snapshot(
        {"difficulty": "hard"},
        {"played_action_probability": 0.10, "label_is_teacher": True},
        {"algorithm_fraction": 0.70, "model_fraction": 0.30},
    )
    assert decision.admitted


def test_growth_never_absorbs_a_shard_an_admitted_snapshot_trained_on(
    tmp_path: Path,
) -> None:
    """A hold-out built from already-trained shards measures memorisation.

    The 2026-08-22 growth absorbed 8 shards that were training files in the
    active snapshot, leaving 87.93% of the hold-out already fit by the model.
    """
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    snapshot_root = tmp_path / "snapshots"
    for index in range(6):
        _write_replay(replay_dir / f"replay_seed_{index}.jsonl", [_entry(index)])

    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(snapshot_root),
        validation_fraction=0.50,
        split_seed=5,
        min_fresh_fraction=0.0,
    )
    _admit_series(manager, replay_dir, 1)
    trained = manager._trained_shard_names()
    assert trained, "the admitted snapshot must report its training shards"

    # Offer fresh, never-trained shards alongside the trained ones.
    for index in range(6):
        _write_replay(
            replay_dir / f"replay_fresh_{index}.jsonl", [_entry(400 + index)])
    files = sorted(replay_dir.glob("replay_*.jsonl"))
    manifest, _keys = manager._ensure_validation(files)

    added = {
        name
        for event in manifest.get("growth_history", [])
        for name in event["added_files"]
    }
    assert added, "growth should still absorb the untrained shards"
    assert not (added & trained), (
        f"growth absorbed trained shards: {sorted(added & trained)}")


def test_holdout_reports_how_much_of_it_is_still_in_the_corpus(
    tmp_path: Path,
) -> None:
    """A hold-out whose shards rotated out must not advertise its target share."""
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    snapshot_root = tmp_path / "snapshots"
    for index in range(4):
        _write_replay(replay_dir / f"replay_seed_{index}.jsonl", [_entry(index)])

    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(snapshot_root),
        validation_fraction=0.25,
        split_seed=5,
        min_fresh_fraction=0.0,
    )
    _admit_series(manager, replay_dir, 1)
    files = sorted(replay_dir.glob("replay_*.jsonl"))
    manifest, _keys = manager._ensure_validation(files)
    held = {record["name"] for record in manifest["files"]}
    assert held

    # Rotate every held shard out of the replay window.
    for path in list(replay_dir.glob("replay_*.jsonl")):
        if path.name in held:
            path.unlink()
    for index in range(8):
        _write_replay(replay_dir / f"replay_after_{index}.jsonl", [_entry(700 + index)])
    files = sorted(replay_dir.glob("replay_*.jsonl"))
    manifest, _keys = manager._ensure_validation(files)
    split = manifest["split"]

    assert split["validation_files_in_corpus"] < split["validation_file_count"]
    assert split["realized_in_corpus_fraction"] < split["realized_file_fraction"]
    # Both ratios stay in range even when held shards outlive the window.
    assert 0.0 <= split["realized_file_fraction"] <= 1.0
    assert 0.0 <= split["realized_in_corpus_fraction"] <= 1.0


def test_growth_resumes_after_the_holdout_rotates_out_of_the_window(
    tmp_path: Path,
) -> None:
    """End-to-end wiring: _grow_validation must pass the *present* count.

    Sized so the two accountings disagree absolutely: all-time accounting sees
    the target already met and grows nothing, present accounting sees a
    hold-out that covers none of the live corpus and reopens every slot.
    """
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    snapshot_root = tmp_path / "snapshots"
    for index in range(8):
        _write_replay(replay_dir / f"replay_a{index}.jsonl", [_entry(index)])

    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(snapshot_root),
        validation_fraction=0.25,
        split_seed=5,
        min_fresh_fraction=0.0,
    )
    files = sorted(replay_dir.glob("replay_*.jsonl"))
    manifest, _keys = manager._ensure_validation(files)
    held = {record["name"] for record in manifest["files"]}
    assert len(held) == 2, "target for 8 files at 25% is 2"

    # Every held shard leaves the window; the corpus size is unchanged.
    for name in held:
        (replay_dir / name).unlink()
    for index in range(len(held)):
        _write_replay(replay_dir / f"replay_b{index}.jsonl", [_entry(900 + index)])

    files = sorted(replay_dir.glob("replay_*.jsonl"))
    assert len(files) == 8
    manifest, _keys = manager._ensure_validation(files)

    history = manifest.get("growth_history", [])
    assert history, (
        "hold-out covers none of the live corpus but growth did not reopen -- "
        "the quota is counting all-time held shards")
    assert manifest["split"]["validation_files_in_corpus"] >= 1


def test_growth_quota_measures_against_holdout_files_still_present(
    tmp_path: Path,
) -> None:
    """The quota must count present hold-out shards, not all-time ones.

    Counting all-time held shards lets a hold-out whose files have rotated out
    of the replay window report its target share while covering none of the
    live corpus -- F3 recurring at 15% magnitude instead of 1.7%.
    """
    manager = CorpusSnapshotManager(
        str(tmp_path / "replay"),
        str(tmp_path / "snapshots"),
        validation_fraction=0.25,
        split_seed=5,
    )
    # 20 files -> target 5. Nine shards were absorbed all-time, but only one
    # still exists in the window, so eight slots must reopen.
    assert manager._validation_growth_quota(
        total_files=20, held=1, candidates=10, all_time_held=9) == 4
    # With every held shard still present the quota is satisfied and closed.
    assert manager._validation_growth_quota(
        total_files=20, held=5, candidates=10, all_time_held=5) == 0
    # The survivor guard still applies.
    assert manager._validation_growth_quota(
        total_files=20, held=0, candidates=1, all_time_held=0) == 0


def test_holdout_growth_is_bounded_by_the_file_ceiling(tmp_path: Path) -> None:
    """Present-held accounting must not grow the manifest once per rotation."""
    from dama.ai.ml.corpus import HOLDOUT_FILE_CEILING

    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    snapshot_root = tmp_path / "snapshots"
    for index in range(4):
        _write_replay(replay_dir / f"replay_seed_{index}.jsonl", [_entry(index)])
    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(snapshot_root),
        validation_fraction=0.25,
        split_seed=5,
        min_fresh_fraction=0.0,
    )
    _admit_series(manager, replay_dir, 1)

    # Repeatedly rotate the whole window so the quota keeps reopening.
    for cycle in range(12):
        for path in list(replay_dir.glob("replay_*.jsonl")):
            path.unlink()
        for index in range(4):
            _write_replay(
                replay_dir / f"replay_c{cycle}_{index}.jsonl",
                [_entry(5000 + cycle * 50 + index)],
            )
        files = sorted(replay_dir.glob("replay_*.jsonl"))
        manifest, _keys = manager._ensure_validation(files)

    target = max(1, round(4 * 0.25))
    assert len(manifest["files"]) <= HOLDOUT_FILE_CEILING * target


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


def test_replay_audit_catches_late_contract_violation():
    from dama.ai.ml import corpus

    valid = _contract_entry(1, "algorithm", "late-check-1")
    invalid = _contract_entry(2, "algorithm", "late-check-2")
    invalid["chosen_index"] = 999

    # Use the real iterator only for the assertion setup below; replacing it
    # keeps this test independent of JSONL formatting and file I/O.
    original = corpus._iter_entry_dicts
    try:
        corpus._iter_entry_dicts = lambda _path: iter((valid, invalid))
        result = corpus.audit_policy_replay_file(
            Path("sentinel.jsonl"), (2, 4, 6, 8))
    finally:
        corpus._iter_entry_dicts = original
    assert result["valid"] is False
    assert result["records"] == 2
    assert result["errors"]["invalid_teacher_index"] == 1


def test_replay_analysis_cache_matches_cold_scan_and_skips_reparse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dama.ai.ml import corpus

    corpus._clear_replay_file_cache()
    shared = _state(0)
    first = _entry(0, state=shared)
    first["game_id"] = "cycle-000001-algorithm-000001"
    first["trajectory_source"] = "algorithm"
    second = _entry(1, forced=True)
    second["game_id"] = "cycle-000001-algorithm-000002"
    left = tmp_path / "replay_left.jsonl"
    _write_replay(left, [first, second])

    repeated = _entry(0, state=shared)
    repeated["game_id"] = "cycle-000002-current-model-000001"
    repeated["trajectory_source"] = "current_model"
    unique = _entry(2)
    unique["game_id"] = "cycle-000002-current-model-000002"
    right = tmp_path / "replay_right.jsonl"
    _write_replay(right, [repeated, unique])

    previous = {canonical_state_key(shared)}
    cold = corpus.analyze_replay_files([left, right], previous)

    original_iterator = corpus._iter_entry_dicts

    def fail_if_reparsed(_path):
        raise AssertionError("unchanged replay file was reparsed")

    monkeypatch.setattr(corpus, "_iter_entry_dicts", fail_if_reparsed)
    warm = corpus.analyze_replay_files([left, right], previous)

    assert warm == cold
    monkeypatch.setattr(corpus, "_iter_entry_dicts", original_iterator)


def test_replay_analysis_cache_invalidates_on_file_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dama.ai.ml import corpus

    corpus._clear_replay_file_cache()
    path = tmp_path / "replay_mutating.jsonl"
    _write_replay(path, [_entry(0)])
    cold_before = corpus.analyze_replay_files([path])

    _write_replay(path, [_entry(0), _entry(20, forced=True)])
    original_iterator = corpus._iter_entry_dicts
    parse_count = 0

    def counted_iterator(file_path):
        nonlocal parse_count
        parse_count += 1
        yield from original_iterator(file_path)

    monkeypatch.setattr(corpus, "_iter_entry_dicts", counted_iterator)
    warm_after_mutation = corpus.analyze_replay_files([path])
    assert parse_count == 1

    monkeypatch.setattr(corpus, "_iter_entry_dicts", original_iterator)
    corpus._clear_replay_file_cache()
    cold_after_mutation = corpus.analyze_replay_files([path])

    assert warm_after_mutation == cold_after_mutation
    assert warm_after_mutation != cold_before


def test_replay_hash_cache_reuses_and_invalidates_by_file_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dama.ai.ml import corpus

    corpus._clear_replay_file_cache()
    path = tmp_path / "replay_hash.jsonl"
    _write_replay(path, [_entry(0)])
    original_hash = corpus._sha256_file_uncached
    hash_calls = 0

    def counted_hash(file_path):
        nonlocal hash_calls
        hash_calls += 1
        return original_hash(file_path)

    monkeypatch.setattr(corpus, "_sha256_file_uncached", counted_hash)
    first = corpus.replay_file_sha256(path)
    assert corpus.replay_file_sha256(path) == first
    assert hash_calls == 1

    _write_replay(path, [_entry(0), _entry(20)])
    changed = corpus.replay_file_sha256(path)
    assert changed != first
    assert hash_calls == 2


def test_malformed_replay_json_is_never_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dama.ai.ml import corpus

    corpus._clear_replay_file_cache()
    path = tmp_path / "replay_malformed.jsonl"
    path.write_text(json.dumps(_entry(0)) + "\nnot-json\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid replay JSON"):
        corpus.analyze_replay_files([path])

    original_iterator = corpus._iter_entry_dicts
    parse_count = 0

    def counted_iterator(file_path):
        nonlocal parse_count
        parse_count += 1
        yield from original_iterator(file_path)

    monkeypatch.setattr(corpus, "_iter_entry_dicts", counted_iterator)
    with pytest.raises(ValueError, match="Invalid replay JSON"):
        corpus.analyze_replay_files([path])
    assert parse_count == 1


def test_semantically_malformed_replay_is_not_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dama.ai.ml import corpus

    corpus._clear_replay_file_cache()
    path = tmp_path / "replay_semantically_malformed.jsonl"
    path.write_text(
        json.dumps(_entry(0)) + "\n" + json.dumps({"state": {}}) + "\n",
        encoding="utf-8",
    )
    first, _ = corpus.analyze_replay_files([path])
    assert first["malformed_records"] == 1

    original_iterator = corpus._iter_entry_dicts
    parse_count = 0

    def counted_iterator(file_path):
        nonlocal parse_count
        parse_count += 1
        yield from original_iterator(file_path)

    monkeypatch.setattr(corpus, "_iter_entry_dicts", counted_iterator)
    second, _ = corpus.analyze_replay_files([path])
    assert second == first
    assert parse_count == 1


def test_replay_audit_cache_reuses_unchanged_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dama.ai.ml import corpus

    corpus._clear_replay_file_cache()
    path = tmp_path / "replay_audited.jsonl"
    _write_replay(path, [_contract_entry(1, "algorithm", "audit-1")])
    cold = corpus.audit_policy_replay_file(path, (2, 4, 6, 8))

    def fail_if_reparsed(_path):
        raise AssertionError("unchanged replay file was reparsed")

    monkeypatch.setattr(corpus, "_iter_entry_dicts", fail_if_reparsed)
    warm = corpus.audit_policy_replay_file(path, (2, 4, 6, 8))
    assert warm == cold


def test_legacy_replay_audit_fails_fast(monkeypatch):
    from dama.ai.ml import corpus

    legacy = {"state": {}, "legal_moves": []}
    consumed = []

    def sentinel_iterator(_path):
        consumed.append(1)
        yield legacy
        raise AssertionError("legacy audit must stop after first contract error")

    monkeypatch.setattr(corpus, "_iter_entry_dicts", sentinel_iterator)
    result = corpus.audit_policy_replay_file(Path("legacy.jsonl"), (2, 4, 6, 8))
    assert result["valid"] is False
    assert result["records"] == 1
    assert result["errors"] == {"missing_legal_moves": 1}
    assert consumed == [1]


def test_partial_cycle_replay_file_is_rejected_by_per_file_split(
    tmp_path: Path,
) -> None:
    """A killed cycle leaves an off-ratio file; it must not poison the corpus."""
    from dama.ai.ml import corpus

    corpus._clear_replay_file_cache()
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()

    def _cycle(name: str, algorithm: int, model: int, base: int) -> None:
        entries = []
        for index in range(algorithm + model):
            source = "algorithm" if index < algorithm else "current_model"
            entries.append(_contract_entry(base + index, source, f"{name}-{index}"))
        _write_replay(replay_dir / name, entries)

    _cycle("replay_complete_0.jsonl", 7, 3, 0)
    _cycle("replay_complete_1.jsonl", 7, 3, 100)
    # Interrupted cycle: the model trajectories were still running when the
    # process died, so the file holds 7/2 instead of 7/3.
    _cycle("replay_partial.jsonl", 7, 2, 200)

    partial = corpus.audit_policy_replay_file(
        replay_dir / "replay_partial.jsonl", (2, 4, 6, 8))
    assert partial["valid"] is False
    assert partial["errors"] == {"unbalanced_policy_trajectory_split": 1}

    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(tmp_path / "snapshots"),
        validation_fraction=0.50,
        split_seed=3,
        min_fresh_fraction=0.50,
        enforce_policy_contract=True,
        allowed_opening_plies=(2, 4, 6, 8),
    )
    eligible, rejected = manager.eligible_replay_files()
    assert [path.name for path in eligible] == [
        "replay_complete_0.jsonl", "replay_complete_1.jsonl",
    ]
    assert set(rejected) == {"replay_partial.jsonl"}

    # The aggregate contract holds again once the partial file is excluded.
    decision = manager.consider_snapshot(
        {"difficulty": "hard"},
        {"played_action_probability": 0.10},
        {"algorithm_fraction": 0.70, "model_fraction": 0.30},
    )
    assert decision.admitted
    assert decision.metrics["source_game_counts"] == {
        "algorithm": 7, "current_model": 3,
    }


def test_snapshot_load_accepts_windows_written_manifest_paths(tmp_path: Path) -> None:
    """A manifest written by the native-Windows launcher must load on WSL.

    ``str(Path("files") / name)`` emits the *host* separator, so a snapshot or
    hold-out grown on native Windows records ``files\\replay_*.jsonl``.  Read
    back on Linux that is a single filename containing a backslash: every stored
    shard "fails integrity verification" while sitting untouched on disk.  Both
    the training snapshot and the frozen validation manifest are rewritten here
    because ``_grow_validation`` carries old records forward verbatim, so a
    single Windows run leaves the two manifests permanently mixed.
    """
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    for index in range(4):
        _write_replay(replay_dir / f"replay_{index}.jsonl", [_entry(index)])
    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(tmp_path / "snapshots"),
        validation_fraction=0.25,
        split_seed=29,
        min_fresh_fraction=0.50,
    )
    decision = manager.consider_snapshot({}, {}, {})

    for manifest_path in (
        decision.manifest_path,
        manager.validation_manifest_path,
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["files"], manifest_path
        for record in manifest["files"]:
            record["path"] = record["path"].replace("/", "\\")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    train_entries, validation_entries, _ = manager.load_split(
        decision.manifest_path)
    assert train_entries
    assert validation_entries
