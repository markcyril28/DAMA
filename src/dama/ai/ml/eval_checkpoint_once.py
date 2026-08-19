"""Evaluate one checkpoint and emit one compact JSON record to stdout.

This module must live in a real Python file because ModelVsAlgoTester uses
multiprocessing with the spawn start method. Running the same code through
``python -`` makes child processes try to import ``<stdin>``, which fails.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys

from .model_vs_algo import ModelVsAlgoTester


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one Dama checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to model_step_*.pt")
    parser.add_argument("--checkpoint-name", required=True, help="Checkpoint basename")
    parser.add_argument("--step", required=True, type=int, help="Training step")
    parser.add_argument("--num-games", required=True, type=int, help="Games to run")
    parser.add_argument("--difficulty", required=True, help="Algorithmic AI difficulty")
    parser.add_argument(
        "--opponent",
        choices=("algorithm", "random"),
        default="algorithm",
        help="Opponent policy. Random selects uniformly from all legal moves.",
    )
    parser.add_argument("--num-workers", required=True, type=int, help="Parallel test workers")
    parser.add_argument("--max-moves", required=True, type=int, help="Draw after this many moves")
    parser.add_argument("--stats-dir", default="models/test_stats", help="Directory for detailed test JSON")
    # [Pass 109] Without a random opening, every game starts from
    # GameState.initial() and a deterministic argmax model replays the same two
    # games, so --num-games N reports an N-sample measurement of 2 samples.
    parser.add_argument("--opening-plies", default="0,2,4,6,8",
                        help="Comma-separated random opening lengths cycled across "
                             "games; '0' restores the old fixed opening")
    parser.add_argument(
        "--opening-seed",
        default=1234,
        type=int,
        help="Fixed seed identifying the paired opening suite",
    )
    parser.add_argument(
        "--inference-depth",
        default=1,
        type=int,
        choices=(1, 2, 3),
        help="ML inference depth. Depths 2 and 3 require a value head.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    tester = ModelVsAlgoTester(
        model_path=args.checkpoint,
        algo_difficulty=args.difficulty,
        num_workers=args.num_workers,
        max_moves=args.max_moves,
        stats_dir=args.stats_dir,
        opening_plies=tuple(int(x) for x in args.opening_plies.split(",") if x.strip()) or (0,),
        opening_seed=args.opening_seed,
        opponent_type=args.opponent,
        ml_inference_depth=args.inference_depth,
    )

    def progress(done, total, stats):
        if done % 20 == 0 or done == total:
            print(
                f"  [{done}/{total}] match_score={stats.match_score:.2%}",
                file=sys.stderr,
            )

    # ModelVsAlgoTester prints status to stdout. Keep stdout reserved for the
    # final JSONL record so the shell wrapper can append it safely.
    with contextlib.redirect_stdout(sys.stderr):
        stats = tester.run_tests(num_games=args.num_games, callback=progress)

    if stats.total_games != args.num_games:
        print(
            f"ERROR: evaluation completed {stats.total_games}/{args.num_games} games",
            file=sys.stderr,
        )
        return 2

    serialized = stats.to_dict()
    entry = {
        "type": "test_vs_algo",
        "checkpoint": args.checkpoint_name,
        "step": args.step,
        "total_games": stats.total_games,
        "ml_wins": stats.ml_wins,
        "algo_wins": stats.algo_wins,
        "draws": stats.draws,
        "opponent_wins": stats.algo_wins,
        "overall_wdl": serialized["overall_wdl"],
        "ml_as_p1_wdl": serialized["ml_as_p1_wdl"],
        "ml_as_p2_wdl": serialized["ml_as_p2_wdl"],
        "match_score": round(stats.match_score, 6),
        "match_score_ci_95": serialized["match_score_ci_95"],
        "ci_method": serialized["ci_method"],
        "ml_win_rate": round(stats.ml_win_rate, 4),
        "draw_rate": round(stats.draw_rate, 4),
        "ml_as_p1_win_rate": round(stats.ml_as_p1_win_rate, 4),
        "ml_as_p2_win_rate": round(stats.ml_as_p2_win_rate, 4),
        "avg_game_length": round(stats.avg_game_length, 2),
        "algo_difficulty": args.difficulty,
        "opponent_type": stats.opponent_type,
        "opening_seed": stats.opening_seed,
        "opening_plies": stats.opening_plies,
        "opening_suite_id": stats.opening_suite_id,
        "opening_suite_size": stats.opening_suite_size,
        "ml_inference_depth": stats.ml_inference_depth,
        "timestamp": stats.end_time,
    }
    print(json.dumps(entry, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
