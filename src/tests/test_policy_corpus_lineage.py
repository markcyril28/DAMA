"""Rebuilt corpus lineage, versioned hold-out, and the all-time trained ledger.

Merged audit Suggestions 2 and 3: snapshot_v000012 was admitted with a lost
CURRENT pointer, so its recorded 100% freshness was an artifact of an empty
previous-key set, and v13-v16 all descend from it.  The approved disposition is
to preserve those artifacts as evidence and branch a new lineage from v11, the
last valid predecessor, with a versioned replacement hold-out that excludes
every historically trained state.
"""

import json
from pathlib import Path

import pytest

from dama.ai.ml.corpus import CorpusSnapshotManager


TEACHER = {"difficulty": "hard"}
NOISE = {"played_action_probability": 0.10, "label_is_teacher": True}
GENERATION = {"algorithm_fraction": 0.70, "model_fraction": 0.30}


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


def _entry(index: int) -> dict:
    return {
        "state": _state(index),
        "legal_moves": [
            {"path": [[0, 1], [1, 0]], "captures": [], "promotion": False},
            {"path": [[0, 1], [1, 2]], "captures": [], "promotion": False},
        ],
        "chosen_index": 0,
        "result": 0,
        "trajectory_source": "algorithm",
    }


def _write_replay(path: Path, indices) -> None:
    path.write_text(
        "".join(json.dumps(_entry(index), sort_keys=True) + "\n" for index in indices),
        encoding="utf-8",
    )


def _manager(replay_dir: Path, root: Path, **kwargs) -> CorpusSnapshotManager:
    options = {
        "validation_fraction": 0.25,
        "split_seed": 5,
        "min_fresh_fraction": 0.50,
        "grow_holdout": False,
    }
    options.update(kwargs)
    return CorpusSnapshotManager(str(replay_dir), str(root), **options)


def _build_base(
    tmp_path: Path, name: str = "base", first_index: int = 0
) -> tuple[Path, dict]:
    """Admit one snapshot in a preserved namespace and return its manifest.

    ``name``/``first_index`` exist so a test can build a *second*, genuinely
    different base whose fingerprint cannot collide with the first.
    """
    replay_dir = tmp_path / f"{name}_replay"
    replay_dir.mkdir()
    for index in range(4):
        _write_replay(
            replay_dir / f"replay_{name}_{index}.jsonl", [first_index + index])
    manager = _manager(replay_dir, tmp_path / f"{name}_root")
    decision = manager.consider_snapshot(TEACHER, NOISE, GENERATION)
    assert decision.admitted and decision.manifest_path is not None
    return decision.manifest_path, json.loads(
        decision.manifest_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Lineage base
# ---------------------------------------------------------------------------

def test_first_admission_is_gated_against_the_configured_lineage_base(
    tmp_path: Path,
) -> None:
    """The defect being fixed: a namespace with no CURRENT admitted anything.

    Without a base the first admission skips the freshness gate entirely, which
    is how v12 recorded 100% freshness.  With a base configured, the very first
    snapshot of the rebuilt lineage is gated exactly like every later one.
    """
    base_manifest_path, base_manifest = _build_base(tmp_path)
    base_states = base_manifest["metrics"]["post_dedup_unique_state_count"]
    assert base_states > 0

    replay_dir = tmp_path / "new_replay"
    replay_dir.mkdir()
    # Reuse the base corpus verbatim: nothing here is fresh relative to v11.
    for index in range(4):
        _write_replay(replay_dir / f"replay_new_{index}.jsonl", [index])
    manager = _manager(
        replay_dir,
        tmp_path / "new_root",
        lineage_base_manifest=str(base_manifest_path),
        lineage_base_fingerprint=base_manifest["fingerprint"],
    )
    stale = manager.consider_snapshot(TEACHER, NOISE, GENERATION)
    assert not stale.admitted
    assert "below" in stale.reason
    assert stale.metrics["fresh_unique_state_rate"] == pytest.approx(0.0)
    assert not list((tmp_path / "new_root").glob("snapshot_v*"))


def test_admission_from_the_lineage_base_records_verified_ancestry(
    tmp_path: Path,
) -> None:
    base_manifest_path, base_manifest = _build_base(tmp_path)

    replay_dir = tmp_path / "new_replay"
    replay_dir.mkdir()
    for index in range(4):
        _write_replay(replay_dir / f"replay_new_{index}.jsonl", [500 + index])
    manager = _manager(
        replay_dir,
        tmp_path / "new_root",
        lineage_base_manifest=str(base_manifest_path),
        lineage_base_fingerprint=base_manifest["fingerprint"],
        lineage_excluded_fingerprints=["dead" * 16],
    )
    decision = manager.consider_snapshot(TEACHER, NOISE, GENERATION)
    assert decision.admitted
    manifest = json.loads(decision.manifest_path.read_text(encoding="utf-8"))
    assert manifest["previous_fingerprint"] == base_manifest["fingerprint"]
    assert manifest["admission"]["previous_corpus_source"] == "lineage_base"
    assert manifest["lineage_base"]["fingerprint"] == base_manifest["fingerprint"]
    assert manifest["lineage_base"]["version"] == base_manifest["version"]
    assert manifest["lineage_base"]["excluded_fingerprints"] == ["dead" * 16]

    _train, _validation, loaded = manager.load_split()
    assert loaded["lineage_verification"]["enforced"] is True
    assert loaded["lineage_verification"]["base_fingerprint"] == (
        base_manifest["fingerprint"])


def test_lineage_base_fingerprint_mismatch_fails_closed(tmp_path: Path) -> None:
    base_manifest_path, base_manifest = _build_base(tmp_path)
    replay_dir = tmp_path / "new_replay"
    replay_dir.mkdir()
    for index in range(4):
        _write_replay(replay_dir / f"replay_new_{index}.jsonl", [500 + index])
    manager = _manager(
        replay_dir,
        tmp_path / "new_root",
        lineage_base_manifest=str(base_manifest_path),
        lineage_base_fingerprint="0" * 64,
    )
    with pytest.raises(RuntimeError, match="lineage base fingerprint mismatch"):
        manager.consider_snapshot(TEACHER, NOISE, GENERATION)
    assert base_manifest["fingerprint"] != "0" * 64


def test_lineage_base_must_be_configured_as_a_pair(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="configured together"):
        _manager(tmp_path, tmp_path / "root", lineage_base_manifest="somewhere")
    with pytest.raises(ValueError, match="configured together"):
        _manager(tmp_path, tmp_path / "root", lineage_base_fingerprint="a" * 64)


def test_lineage_base_cannot_also_be_excluded(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot also be excluded"):
        _manager(
            tmp_path,
            tmp_path / "root",
            lineage_base_manifest="somewhere",
            lineage_base_fingerprint="a" * 64,
            lineage_excluded_fingerprints=["A" * 64],
        )


def test_load_split_refuses_a_snapshot_with_excluded_ancestry(
    tmp_path: Path,
) -> None:
    """The v12-v16 exclusion, enforced at load rather than only at admission."""
    base_manifest_path, base_manifest = _build_base(tmp_path)
    replay_dir = tmp_path / "new_replay"
    replay_dir.mkdir()
    for index in range(4):
        _write_replay(replay_dir / f"replay_new_{index}.jsonl", [500 + index])
    root = tmp_path / "new_root"
    admitting = _manager(
        replay_dir,
        root,
        lineage_base_manifest=str(base_manifest_path),
        lineage_base_fingerprint=base_manifest["fingerprint"],
    )
    assert admitting.consider_snapshot(TEACHER, NOISE, GENERATION).admitted

    # The same corpus, now read by a run that declares this ancestry defective.
    manifest = json.loads(
        admitting.current_manifest_path().read_text(encoding="utf-8"))
    excluding = _manager(
        replay_dir,
        root,
        lineage_base_manifest=str(base_manifest_path),
        lineage_base_fingerprint=base_manifest["fingerprint"],
        lineage_excluded_fingerprints=[manifest["fingerprint"]],
    )
    with pytest.raises(RuntimeError, match="excluded snapshot fingerprint"):
        excluding.load_split()


def test_load_split_refuses_a_run_that_relaxed_a_recorded_exclusion(
    tmp_path: Path,
) -> None:
    """A snapshot admitted under a stricter policy must not load under a weaker one."""
    base_manifest_path, base_manifest = _build_base(tmp_path)
    replay_dir = tmp_path / "new_replay"
    replay_dir.mkdir()
    for index in range(4):
        _write_replay(replay_dir / f"replay_new_{index}.jsonl", [500 + index])
    root = tmp_path / "new_root"
    strict = _manager(
        replay_dir,
        root,
        lineage_base_manifest=str(base_manifest_path),
        lineage_base_fingerprint=base_manifest["fingerprint"],
        lineage_excluded_fingerprints=["dead" * 16],
    )
    assert strict.consider_snapshot(TEACHER, NOISE, GENERATION).admitted

    relaxed = _manager(
        replay_dir,
        root,
        lineage_base_manifest=str(base_manifest_path),
        lineage_base_fingerprint=base_manifest["fingerprint"],
    )
    with pytest.raises(RuntimeError, match="no longer excludes"):
        relaxed.load_split()


def test_load_split_refuses_a_snapshot_that_records_no_lineage_base(
    tmp_path: Path,
) -> None:
    """Historical, unstamped data cannot be adopted by an enforcing run."""
    base_manifest_path, base_manifest = _build_base(tmp_path)
    replay_dir = tmp_path / "new_replay"
    replay_dir.mkdir()
    for index in range(4):
        _write_replay(replay_dir / f"replay_new_{index}.jsonl", [500 + index])
    root = tmp_path / "new_root"
    # Admitted with no lineage base at all -- the historical behaviour.
    plain = _manager(replay_dir, root)
    assert plain.consider_snapshot(TEACHER, NOISE, GENERATION).admitted

    enforcing = _manager(
        replay_dir,
        root,
        lineage_base_manifest=str(base_manifest_path),
        lineage_base_fingerprint=base_manifest["fingerprint"],
    )
    with pytest.raises(RuntimeError, match="records no lineage base"):
        enforcing.load_split()


def test_load_split_refuses_a_snapshot_from_a_different_base(
    tmp_path: Path,
) -> None:
    """The foreign-base branch, distinct from the missing-record branch above.

    A snapshot stamped with base A must not load under a run that pins base B.
    Both bases are real, integrity-verified snapshots, so the refusal comes
    from the fingerprint comparison and not from a malformed record.
    """
    base_a_path, base_a = _build_base(tmp_path, "base_a")
    base_b_path, base_b = _build_base(tmp_path, "base_b", first_index=100)
    assert base_a["fingerprint"] != base_b["fingerprint"]

    replay_dir = tmp_path / "new_replay"
    replay_dir.mkdir()
    for index in range(4):
        _write_replay(replay_dir / f"replay_new_{index}.jsonl", [500 + index])
    root = tmp_path / "new_root"
    admitting = _manager(
        replay_dir,
        root,
        lineage_base_manifest=str(base_a_path),
        lineage_base_fingerprint=base_a["fingerprint"],
    )
    decision = admitting.consider_snapshot(TEACHER, NOISE, GENERATION)
    assert decision.admitted
    stamped = json.loads(decision.manifest_path.read_text(encoding="utf-8"))
    assert stamped["lineage_base"]["fingerprint"] == base_a["fingerprint"]

    foreign = _manager(
        replay_dir,
        root,
        lineage_base_manifest=str(base_b_path),
        lineage_base_fingerprint=base_b["fingerprint"],
    )
    with pytest.raises(RuntimeError, match="unapproved lineage base"):
        foreign.load_split()


def test_load_split_refuses_a_run_that_declares_no_lineage_at_all(
    tmp_path: Path,
) -> None:
    """Dropping the whole ``lineage:`` block must not be an escape hatch.

    Enforcement is opt-in, so a config with no lineage block is unenforced --
    but that has to mean *unstamped* data.  Otherwise the neighbouring
    relaxed-exclusion refusal is trivially evaded: a run cannot load a
    strictly-admitted snapshot by dropping one excluded fingerprint, so it
    would simply drop the base as well and skip every check at once.
    """
    base_manifest_path, base_manifest = _build_base(tmp_path)
    replay_dir = tmp_path / "new_replay"
    replay_dir.mkdir()
    for index in range(4):
        _write_replay(replay_dir / f"replay_new_{index}.jsonl", [500 + index])
    root = tmp_path / "new_root"
    strict = _manager(
        replay_dir,
        root,
        lineage_base_manifest=str(base_manifest_path),
        lineage_base_fingerprint=base_manifest["fingerprint"],
        lineage_excluded_fingerprints=["dead" * 16],
    )
    assert strict.consider_snapshot(TEACHER, NOISE, GENERATION).admitted

    unenforced = _manager(replay_dir, root)
    with pytest.raises(RuntimeError, match="does not declare"):
        unenforced.load_split()


def test_load_split_stays_unenforced_for_unstamped_legacy_snapshots(
    tmp_path: Path,
) -> None:
    """The preserved wd1e4/policy_distillation roots record no lineage base.

    Refusing stamped data under an unenforced run must not turn into refusing
    the historical namespaces those runs still legitimately read.
    """
    replay_dir = tmp_path / "legacy_replay"
    replay_dir.mkdir()
    for index in range(4):
        _write_replay(replay_dir / f"replay_legacy_{index}.jsonl", [700 + index])
    root = tmp_path / "legacy_root"
    legacy = _manager(replay_dir, root)
    decision = legacy.consider_snapshot(TEACHER, NOISE, GENERATION)
    assert decision.admitted
    assert "lineage_base" not in json.loads(
        decision.manifest_path.read_text(encoding="utf-8"))

    train, _validation, manifest = legacy.load_split()
    assert train
    assert manifest["lineage_verification"] == {"enforced": False}


def test_lineage_base_is_stamped_on_every_snapshot_so_pruning_cannot_erase_it(
    tmp_path: Path,
) -> None:
    base_manifest_path, base_manifest = _build_base(tmp_path)
    replay_dir = tmp_path / "new_replay"
    replay_dir.mkdir()
    for index in range(4):
        _write_replay(replay_dir / f"replay_new_{index}.jsonl", [500 + index])
    manager = _manager(
        replay_dir,
        tmp_path / "new_root",
        lineage_base_manifest=str(base_manifest_path),
        lineage_base_fingerprint=base_manifest["fingerprint"],
        max_retained_snapshots=1,
    )
    first = manager.consider_snapshot(TEACHER, NOISE, GENERATION)
    assert first.admitted
    first_states = json.loads(first.manifest_path.read_text(encoding="utf-8"))[
        "metrics"]["post_dedup_unique_state_count"]
    for offset in range(first_states * 2):
        _write_replay(replay_dir / f"replay_grow_{offset}.jsonl", [9000 + offset])
    second = manager.consider_snapshot(TEACHER, NOISE, GENERATION)
    assert second.admitted

    root = tmp_path / "new_root"
    assert len(list(root.glob("snapshot_v*"))) == 1, "retention must have pruned"
    surviving = json.loads(second.manifest_path.read_text(encoding="utf-8"))
    # The first snapshot -- the only one whose previous_fingerprint pointed at
    # the base -- is gone, yet the ancestry claim is still checkable.
    assert surviving["previous_fingerprint"] != base_manifest["fingerprint"]
    assert surviving["lineage_base"]["fingerprint"] == base_manifest["fingerprint"]
    assert manager.load_split()[2]["lineage_verification"]["enforced"] is True


# ---------------------------------------------------------------------------
# Versioned hold-out replacement
# ---------------------------------------------------------------------------

def test_validation_generation_two_lands_beside_the_superseded_split(
    tmp_path: Path,
) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    for index in range(4):
        _write_replay(replay_dir / f"replay_{index}.jsonl", [index])
    root = tmp_path / "root"

    first = _manager(replay_dir, root)
    assert first.consider_snapshot(TEACHER, NOISE, GENERATION).admitted
    assert (root / "validation" / "manifest.json").is_file()

    rebuilt = _manager(replay_dir, root, validation_split_version=2)
    assert rebuilt.validation_manifest_path == root / "validation_v2" / "manifest.json"
    for offset in range(40):
        _write_replay(replay_dir / f"replay_new_{offset}.jsonl", [700 + offset])
    assert rebuilt.consider_snapshot(TEACHER, NOISE, GENERATION).admitted
    assert (root / "validation_v2" / "manifest.json").is_file()
    # The superseded artifact survives untouched as evidence.
    assert (root / "validation" / "manifest.json").is_file()
    rebuilt_split = json.loads(
        (root / "validation_v2" / "manifest.json").read_text(encoding="utf-8")
    )["split"]
    assert rebuilt_split["version"] == 2


def test_a_superseded_hold_out_generation_is_never_silently_reused(
    tmp_path: Path,
) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    for index in range(4):
        _write_replay(replay_dir / f"replay_{index}.jsonl", [index])
    root = tmp_path / "root"
    assert _manager(replay_dir, root).consider_snapshot(
        TEACHER, NOISE, GENERATION).admitted

    # Point generation 2 at the generation-1 directory, as a rename would.
    (root / "validation").rename(root / "validation_v2")
    rebuilt = _manager(replay_dir, root, validation_split_version=2)
    with pytest.raises(RuntimeError, match="generation 1"):
        rebuilt.consider_snapshot(TEACHER, NOISE, GENERATION)


def test_realized_hold_out_share_is_persisted_even_without_growth(
    tmp_path: Path,
) -> None:
    """F3 sub-defect: a decay-only pass printed the recalculation but the
    manifest on disk kept advertising the share it had at its last growth."""
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    for index in range(4):
        _write_replay(replay_dir / f"replay_{index}.jsonl", [index])
    root = tmp_path / "root"
    manager = _manager(replay_dir, root)
    assert manager.consider_snapshot(TEACHER, NOISE, GENERATION).admitted

    manifest_path = root / "validation" / "manifest.json"
    before = json.loads(manifest_path.read_text(encoding="utf-8"))["split"]
    assert before["source_file_count"] == 4

    for offset in range(20):
        _write_replay(replay_dir / f"replay_more_{offset}.jsonl", [800 + offset])
    manager.consider_snapshot(TEACHER, NOISE, GENERATION)

    after = json.loads(manifest_path.read_text(encoding="utf-8"))["split"]
    assert after["source_file_count"] == 24
    assert after["realized_file_fraction"] < before["realized_file_fraction"]


# ---------------------------------------------------------------------------
# All-time trained ledger
# ---------------------------------------------------------------------------

def test_ledger_seeds_from_preserved_roots_and_refuses_to_re_hold_a_trained_shard(
    tmp_path: Path,
) -> None:
    base_manifest_path, base_manifest = _build_base(tmp_path)
    base_root = base_manifest_path.parent.parent
    trained_names = {record["name"] for record in base_manifest["files"]}
    assert trained_names

    replay_dir = tmp_path / "new_replay"
    replay_dir.mkdir()
    # Same shard names as the base corpus: these were demonstrably trained on.
    for index in range(4):
        _write_replay(replay_dir / f"replay_base_{index}.jsonl", [600 + index])
    for index in range(4):
        _write_replay(replay_dir / f"replay_clean_{index}.jsonl", [700 + index])

    manager = _manager(
        replay_dir,
        tmp_path / "new_root",
        trained_ledger_enabled=True,
        trained_ledger_seed_roots=[str(base_root)],
    )
    assert trained_names <= manager.trained_ledger_shard_names()
    assert manager.trained_ledger_state_keys()

    assert manager.consider_snapshot(TEACHER, NOISE, GENERATION).admitted
    held = {
        record["name"] for record in json.loads(
            manager.validation_manifest_path.read_text(encoding="utf-8"))["files"]
    }
    assert held
    assert held.isdisjoint(trained_names)


def test_ledger_records_each_admission_and_survives_snapshot_pruning(
    tmp_path: Path,
) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    for index in range(4):
        _write_replay(replay_dir / f"replay_{index}.jsonl", [index])
    root = tmp_path / "root"
    manager = _manager(
        replay_dir, root, trained_ledger_enabled=True, max_retained_snapshots=1)
    first = manager.consider_snapshot(TEACHER, NOISE, GENERATION)
    assert first.admitted
    first_names = manager.trained_ledger_shard_names()
    assert first_names

    first_states = json.loads(first.manifest_path.read_text(encoding="utf-8"))[
        "metrics"]["post_dedup_unique_state_count"]
    for offset in range(first_states * 2):
        _write_replay(replay_dir / f"replay_grow_{offset}.jsonl", [9000 + offset])
    assert manager.consider_snapshot(TEACHER, NOISE, GENERATION).admitted
    assert len(list(root.glob("snapshot_v*"))) == 1

    # A fresh manager reads the ledger off disk; the pruned snapshot's shards
    # are still known to have been trained on.
    reopened = _manager(replay_dir, root, trained_ledger_enabled=True)
    assert first_names <= reopened.trained_ledger_shard_names()


def test_load_split_removes_hold_out_states_the_ledger_says_were_trained(
    tmp_path: Path,
) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    for index in range(4):
        _write_replay(replay_dir / f"replay_{index}.jsonl", [index])
    root = tmp_path / "root"
    manager = _manager(replay_dir, root, trained_ledger_enabled=True)
    assert manager.consider_snapshot(TEACHER, NOISE, GENERATION).admitted

    _train, validation, manifest = manager.load_split()
    leakage = manifest["validation_leakage"]
    assert leakage["ledger_enabled"] is True
    assert leakage["all_time_trained_state_count"] > 0
    assert leakage["retained_validation_entry_count"] == len(validation)

    # Force the exact defect: declare every held state as historically trained.
    held_keys = set()
    validation_manifest = json.loads(
        manager.validation_manifest_path.read_text(encoding="utf-8"))
    from dama.ai.ml.corpus import _merge_state_keys_file, canonical_state_key
    for record in validation_manifest["files"]:
        path = manager.validation_manifest_path.parent / record["path"]
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                held_keys.add(canonical_state_key(json.loads(line)["state"]))
    assert held_keys
    _merge_state_keys_file(root / "ledger" / "trained_state_keys.txt.gz", held_keys)
    reopened = _manager(replay_dir, root, trained_ledger_enabled=True)
    _train2, validation2, manifest2 = reopened.load_split()
    assert validation2 == []
    assert manifest2["validation_leakage"]["removed_validation_state_count"] == len(
        held_keys)


def test_ledger_is_off_by_default_so_preserved_namespaces_are_never_written(
    tmp_path: Path,
) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    for index in range(4):
        _write_replay(replay_dir / f"replay_{index}.jsonl", [index])
    root = tmp_path / "root"
    manager = _manager(replay_dir, root)
    assert manager.consider_snapshot(TEACHER, NOISE, GENERATION).admitted
    assert not (root / "ledger").exists()
    assert manager.trained_ledger_shard_names() == set()
    assert manager.load_split()[2]["validation_leakage"]["ledger_enabled"] is False


def test_state_key_fingerprints_are_stable_and_streamed_merges_are_lossless(
    tmp_path: Path,
) -> None:
    """The ledger holds 64-bit fingerprints in memory, full keys on disk."""
    from dama.ai.ml.corpus import (
        _merge_state_keys_file,
        _read_state_keys,
        _state_key_fingerprint,
    )

    path = tmp_path / "keys.txt.gz"
    first = {f"{index:064x}" for index in range(0, 40, 2)}
    second = {f"{index:064x}" for index in range(0, 40, 3)}
    assert _merge_state_keys_file(path, first) == len(first)
    added = _merge_state_keys_file(path, second)
    assert added == len(second - first)
    assert _read_state_keys(path) == first | second
    # Re-merging identical content adds nothing and changes nothing.
    before = path.read_bytes()
    assert _merge_state_keys_file(path, second) == 0
    assert _read_state_keys(path) == first | second

    key = "a" * 64
    assert _state_key_fingerprint(key) == int("a" * 16, 16)
    assert _state_key_fingerprint(key) == _state_key_fingerprint(key)


def test_rebuilt_hold_out_reaches_the_approved_share_of_a_sixty_file_corpus(
    tmp_path: Path,
) -> None:
    """Audit Suggestion 2 exit criterion: at least 9 of 60 files live.

    The generation-1 artifact held 27 append-only files but only 3 of the 60
    live ones, because growth was measured against an all-time count that had
    already hit its ceiling. Generation 2 starts clean and must actually
    deliver the approved 15% against today's corpus.
    """
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    root = tmp_path / "root"
    manager = CorpusSnapshotManager(
        str(replay_dir),
        str(root),
        validation_fraction=0.15,
        split_seed=20260819,
        min_fresh_fraction=0.50,
        grow_holdout=True,
        validation_split_version=2,
        trained_ledger_enabled=True,
    )

    # A rolling 60-file corpus, grown the way self-play grows it.
    written = 0
    for cycle in range(6):
        for _ in range(10):
            _write_replay(
                replay_dir / f"replay_{written:03d}.jsonl",
                [10_000 + written * 5 + offset for offset in range(5)],
            )
            written += 1
        manager.consider_snapshot(TEACHER, NOISE, GENERATION)

    assert written == 60
    live_files = {path.name for path in manager.replay_files()}
    assert len(live_files) == 60
    held = json.loads(
        manager.validation_manifest_path.read_text(encoding="utf-8"))
    held_names = {record["name"] for record in held["files"]}
    live_held = held_names & live_files

    target = round(60 * 0.15)
    assert target == 9
    assert len(live_held) >= target, (
        f"hold-out delivered {len(live_held)} of 60 live files, below the "
        f"approved {target}")
    assert held["split"]["validation_files_in_corpus"] >= target
    assert held["split"]["realized_in_corpus_fraction"] >= 0.15
    # Whole-file hold-out is preserved: nothing held is also a training shard.
    current = json.loads(
        manager.current_manifest_path().read_text(encoding="utf-8"))
    assert {record["name"] for record in current["files"]}.isdisjoint(held_names)
