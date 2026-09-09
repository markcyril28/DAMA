"""Snapshot parsing must preserve text JSONL semantics across read boundaries."""

import json
from pathlib import Path

import pytest

from dama.ai.ml import corpus


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
@pytest.mark.parametrize("terminated", [False, True])
def test_replay_text_reader_preserves_long_unicode_records(
    tmp_path: Path, newline: str, terminated: bool,
) -> None:
    entries = [
        {"text": "á棋😀\u2028\u2029" * 220_000, "index": 0},
        {"text": "last", "index": 1},
    ]
    # Unicode whitespace-only lines are skipped by the original text reader.
    lines = ["", "\u2003", json.dumps(entries[0], ensure_ascii=False),
             "  ", json.dumps(entries[1])]
    payload = newline.join(lines) + (newline if terminated else "")
    path = tmp_path / "replay.jsonl"
    path.write_bytes(payload.encode("utf-8"))

    assert list(corpus._iter_entry_dicts(path)) == entries


@pytest.mark.parametrize(
    "invalid, message",
    [("{broken", "Invalid replay JSON"), ("[]", "Replay entry is not an object")],
)
def test_replay_text_reader_reports_first_error_after_long_record(
    tmp_path: Path, invalid: str, message: str,
) -> None:
    entry = {"text": "x" * (2 * 1024 * 1024 + 17)}
    path = tmp_path / "invalid.jsonl"
    path.write_text(
        "\n" + json.dumps(entry) + "\r\n\n" + invalid + "\n{later",
        encoding="utf-8",
    )
    records = corpus._iter_entry_dicts(path)

    assert next(records) == entry
    with pytest.raises(ValueError) as error:
        next(records)
    assert str(error.value) == f"{message} at {path}:4"


@pytest.mark.parametrize("payload", [b"", b"\n\r\n\r  \n"])
def test_replay_text_reader_accepts_empty_input(tmp_path: Path, payload: bytes) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_bytes(payload)

    assert list(corpus._iter_entry_dicts(path)) == []
