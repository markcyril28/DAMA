"""Pin corpus key bytes and boundary failures across canonicalization changes."""

import hashlib

import pytest

from dama.ai.ml import corpus


@pytest.mark.parametrize("turn", [1, 2])
def test_canonical_payload_preserves_four_bitboard_layout(turn):
    pieces = {
        "p1_men": [[0, 0], [7, 7], [0, 0]],
        "p1_kings": [[4, 3]],
        "p2_men": [[1, 2]],
        "p2_kings": [[5, 6]],
    }
    if turn == 2:
        pieces = {
            "p1_men": [[6, 5]],
            "p1_kings": [[2, 1]],
            "p2_men": [[7, 7], [0, 0], [7, 7]],
            "p2_kings": [[3, 4]],
        }
    state = dict(pieces, turn=turn, move_count=99)
    header = (
        f"{corpus.CANONICAL_RULES_ID}|encoding={corpus.ENCODING_VERSION}|"
    ).encode("ascii")
    expected = header + bytes.fromhex(
        "8000000000000001 0000000800000000 "
        "0000000000000400 0000400000000000"
    )
    assert corpus.canonical_state_payload(state) == expected
    assert corpus.canonical_state_key(state) == hashlib.sha256(expected).hexdigest()


@pytest.mark.parametrize("field", ["p1_men", "p1_kings", "p2_men", "p2_kings"])
def test_canonical_payload_rejects_bitboard_overflow(field):
    with pytest.raises(OverflowError):
        corpus.canonical_state_payload({"turn": 1, field: [[8, 0]]})


def test_canonical_payload_keeps_negative_shift_failure():
    with pytest.raises(ValueError, match="negative shift count"):
        corpus.canonical_state_payload({"turn": 1, "p1_men": [[-1, 0]]})


def test_canonical_payload_keeps_header_bound_to_encoding(monkeypatch):
    monkeypatch.setattr(corpus, "ENCODING_VERSION", "contract-test")
    expected = f"{corpus.CANONICAL_RULES_ID}|encoding=contract-test|".encode("ascii")
    assert corpus.canonical_state_payload({}) == expected + bytes(32)
