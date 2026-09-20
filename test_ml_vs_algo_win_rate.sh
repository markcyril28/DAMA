#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# ML MODEL VS ALGORITHM - OVERALL WIN RATE TESTER
# =============================================================================
# Plays one ML checkpoint against the algorithmic AI and reports the metric
# plot_training.py charts as "ML Model vs Algorithm - Overall Win Rate": W/D/L,
# overall win rate, win rate as each side, and the draw-as-half match score
# with its Wilson 95% interval, per difficulty plus a pooled overall row.
#
# Every run is balanced (exactly half the games on each side) and paired over
# the config's testing.opening_plies / testing.opening_seed suite. The suite id
# also depends on games per side, so at the default testing.num_games (100) an
# easy run replays the acceptance report's easy suite game for game; step
# 452000 reproduced its 0/1/99 exactly. eval_checkpoints.sh does not pass that
# suite.
#
# All settings are the variables below; the script takes no arguments.
# TEST_ALL_MODELS sweeps every checkpoint in the run's checkpoint directory, so
# the dashboard's win-rate panel gets a curve over training steps instead of a
# single point. SKIP_ALREADY_TESTED resumes such a sweep.
# Outputs never touch trainer-owned files (training_stats_*.json, logs/<ns>/)
# or eval_checkpoints.sh's results, whose dedupe keys on checkpoint name.
#
# Dashboard contract, shared with plot_training.py: every record it plots is a
# line of models/test_stats/<ns>/win_rate_tester/results.jsonl stamped
# "tester": "test_ml_vs_algo_win_rate.sh", carrying step, step_known,
# algo_difficulty, opponent_type, ml_win_rate and timestamp. It plots one
# series per (results file, difficulty), one point per step, newest timestamp
# winning; a record with step_known false is not plottable and is skipped.
# Changing that file name, those keys or that location changes what the panel
# can show, so change both scripts together.
# Exit codes: 0 ok, 1 settings/preflight error, 2 evaluation failure,
#             3 overall win rate below MIN_WIN_RATE.
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIG SELECTION
# -----------------------------------------------------------------------------
TEST_CONFIG=""                   # Training YAML to read paths/testing from.
                                 # Empty = the TRAINING_CONFIG selected in local_train.sh.

# -----------------------------------------------------------------------------
# MODEL SELECTION — TEST_ALL_MODELS wins over MODEL_PATH, which wins over MODEL_ALIAS
# -----------------------------------------------------------------------------
TEST_ALL_MODELS=true            # true = test every model_step_*.pt in MODEL_DIR, oldest step
                                 # first, so the dashboard gets a win-rate curve over steps.
MODEL_DIR=""                     # Where TEST_ALL_MODELS looks (empty = the config's paths.checkpoint_dir)
SKIP_ALREADY_TESTED=true         # true = skip a step already recorded in RESULTS_FILE for the same
                                 # difficulty, game count and opening suite, so an interrupted
                                 # sweep resumes. false = measure every model again.
OVERWRITE_PREVIOUS=false         # true = drop the earlier records for each (step, difficulty) this
                                 # run measures, so RESULTS_FILE keeps one result per point on the
                                 # dashboard instead of a growing history. Turns re-testing on, so
                                 # it overrides SKIP_ALREADY_TESTED.
MODEL_ALIAS="latest"             # latest | promoted | accepted (the config's paths.<alias>_model)
MODEL_PATH=""                    # Specific checkpoint, e.g.
                                 # "models/checkpoints_policy_distillation_recovery_c174k/model_step_452000.pt"

# -----------------------------------------------------------------------------
# TEST SETTINGS — Empty = use the config's value
# -----------------------------------------------------------------------------
DIFFICULTIES=""                  # Comma list of easy, medium, hard, super_hard, or "all"
                                 # (empty = testing.difficulty)
NUM_GAMES="100"                     # Games per difficulty, must be even (empty = testing.num_games)
NUM_WORKERS=4                    # Parallel game workers
MAX_MOVES=""                     # Draw after this many moves (empty = selfplay.max_moves_per_game)
OPENING_PLIES=""                 # Comma list of random opening lengths (empty = testing.opening_plies)
OPENING_SEED=""                  # Opening suite seed (empty = testing.opening_seed)
INFERENCE_DEPTH=""               # ML inference depth 1-3 (empty = testing.inference_depth)
MIN_WIN_RATE=""                  # Pass/fail gate in percent, 0-100: exit 3 when the pooled
                                 # overall win rate is below it. Empty = report only. Under
                                 # TEST_ALL_MODELS it gates the pooled rate over every model.

# -----------------------------------------------------------------------------
# ENVIRONMENT AND OUTPUT
# -----------------------------------------------------------------------------
CONDA_ENV="dama"                 # Conda env that provides the CPython 3.11 interpreter
DAMA_PYTHON=""                   # Explicit interpreter path (empty = resolve CONDA_ENV)
STATS_DIR=""                     # Per-run detail JSON (empty = models/test_stats/<ns>/win_rate_tester)
RESULTS_FILE=""                  # One JSONL line per model and difficulty (empty = $STATS_DIR/results.jsonl).
                                 # plot_training.py only finds a file named results.jsonl somewhere under
                                 # models/test_stats; anywhere else the run is measured but never plotted,
                                 # and the script warns when these settings put it there.

# =============================================================================
# END OF PARAMETERS - Do not edit below this line
# =============================================================================

ALL_DIFFICULTIES=(easy medium hard super_hard)

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

die() {
    log "ERROR: $*"
    exit 1
}

if [[ $# -gt 0 ]]; then
    die "This script takes no arguments; edit the variables at the top of $(basename "$0")."
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# A copy edited outside the repository keeps the caller's project root, as
# local_train.sh does for its smoke copies.
if [[ ! -f "$PROJECT_DIR/src/dama/ai/ml/eval_checkpoint_once.py" ]]; then
    _caller_dir="$(pwd -P)"
    if [[ -f "$_caller_dir/src/dama/ai/ml/eval_checkpoint_once.py" && -f "$_caller_dir/local_train.sh" ]]; then
        PROJECT_DIR="$_caller_dir"
    else
        die "Could not resolve the Dama project root from $PROJECT_DIR or $_caller_dir."
    fi
fi

# ========================== CONFIG ===========================================

CONFIG_FILE="$TEST_CONFIG"
if [[ -z "$CONFIG_FILE" ]]; then
    CONFIG_FILE="$({
        awk -F'"' '/^TRAINING_CONFIG="/ { selected = $2 } END { print selected }' \
            "$PROJECT_DIR/local_train.sh"
    } 2>/dev/null)"
    [[ -n "$CONFIG_FILE" ]] || die "Could not resolve the active TRAINING_CONFIG from local_train.sh."
fi
[[ "$CONFIG_FILE" = /* ]] || CONFIG_FILE="$PROJECT_DIR/$CONFIG_FILE"
[[ -f "$CONFIG_FILE" ]] || die "TEST_CONFIG not found: $CONFIG_FILE"
CONFIG_FILE="$(readlink -f "$CONFIG_FILE")"

_yaml_scalar() {
    # _yaml_scalar <file> <top-level-section> <key>
    awk -v section="$2" -v key="$3" '
        /^[^[:space:]#]/ { in_section = ($0 ~ "^"section":[[:space:]]*$"); next }
        in_section && $0 ~ "^[[:space:]]+"key":[[:space:]]*" {
            line = $0
            sub("^[[:space:]]+"key":[[:space:]]*", "", line)
            sub(/[[:space:]]*#.*$/, "", line)
            gsub(/^"|"$|^'"'"'|'"'"'$/, "", line)
            print line
            exit
        }
    ' "$1"
}
_abs_project_path() {
    case "$1" in
        /*) printf '%s' "$1" ;;
        *) printf '%s/%s' "$PROJECT_DIR" "$1" ;;
    esac
}
_rel_project_path() {
    printf '%s' "${1#"$PROJECT_DIR"/}"
}

OUTPUT_NAMESPACE="$(_yaml_scalar "$CONFIG_FILE" paths policy_output_namespace)"
if [[ -n "$OUTPUT_NAMESPACE" && "$OUTPUT_NAMESPACE" == *[!A-Za-z0-9._-]* ]]; then
    die "Unsafe paths.policy_output_namespace in $CONFIG_FILE: $OUTPUT_NAMESPACE"
fi

_is_bool() { [[ "$1" == "true" || "$1" == "false" ]]; }
_is_bool "$TEST_ALL_MODELS" || die "TEST_ALL_MODELS must be true or false (got: '$TEST_ALL_MODELS')"
_is_bool "$SKIP_ALREADY_TESTED" || die "SKIP_ALREADY_TESTED must be true or false (got: '$SKIP_ALREADY_TESTED')"
_is_bool "$OVERWRITE_PREVIOUS" || die "OVERWRITE_PREVIOUS must be true or false (got: '$OVERWRITE_PREVIOUS')"
if [[ "$OVERWRITE_PREVIOUS" == true && "$SKIP_ALREADY_TESTED" == true ]]; then
    # Skipping would leave nothing to overwrite, so the explicit request wins.
    log "NOTE: OVERWRITE_PREVIOUS=true re-measures every selected model (SKIP_ALREADY_TESTED ignored)."
    SKIP_ALREADY_TESTED=false
fi

# Models: TEST_ALL_MODELS sweeps a checkpoint directory; otherwise MODEL_PATH,
# then MODEL_ALIAS from the config, then models/latest.pt for a legacy config
# without a latest_model path.
MODEL_FILES=()
if [[ "$TEST_ALL_MODELS" == true ]]; then
    _model_dir="$MODEL_DIR"
    if [[ -z "$_model_dir" ]]; then
        _model_dir="$(_yaml_scalar "$CONFIG_FILE" paths checkpoint_dir)"
        [[ -n "$_model_dir" ]] || die "paths.checkpoint_dir is missing from $CONFIG_FILE; set MODEL_DIR"
    fi
    _model_dir="$(_abs_project_path "$_model_dir")"
    [[ -d "$_model_dir" ]] || die "MODEL_DIR is not a directory: $_model_dir"
    # sort -V orders model_step_9000 before model_step_10000, so the sweep runs
    # oldest step first and the dashboard curve is built in training order.
    while IFS= read -r _found; do
        [[ -n "$_found" ]] && MODEL_FILES+=("$_found")
    done < <(find "$_model_dir" -maxdepth 1 -type f -name 'model_step_*.pt' -print | sort -V)
    (( ${#MODEL_FILES[@]} > 0 )) || die "No model_step_*.pt checkpoints in $_model_dir"
else
    if [[ -n "$MODEL_PATH" ]]; then
        _model_arg="$MODEL_PATH"
    else
        case "$MODEL_ALIAS" in
            latest|promoted|accepted) ;;
            *) die "MODEL_ALIAS must be latest, promoted, or accepted (got: '$MODEL_ALIAS')" ;;
        esac
        _model_arg="$(_yaml_scalar "$CONFIG_FILE" paths "${MODEL_ALIAS}_model")"
        if [[ -z "$_model_arg" ]]; then
            [[ "$MODEL_ALIAS" == "latest" ]] || die "paths.${MODEL_ALIAS}_model is missing from $CONFIG_FILE"
            _model_arg="models/latest.pt"
        fi
    fi
    _model_file="$(_abs_project_path "$_model_arg")"
    if [[ ! -f "$_model_file" ]]; then
        if [[ -z "$MODEL_PATH" ]]; then
            die "Model not found: $_model_file (MODEL_ALIAS=$MODEL_ALIAS has not been published for this run yet)"
        fi
        die "Model not found: $_model_file"
    fi
    MODEL_FILES=("$_model_file")
fi

# Empty settings follow the trainer's testing block; legacy configs fall back
# to the trainer's own dataclass defaults (test_opening_plies / test_opening_seed).
_cfg_games="$(_yaml_scalar "$CONFIG_FILE" testing num_games)"
_cfg_difficulty="$(_yaml_scalar "$CONFIG_FILE" testing difficulty)"
_cfg_plies="$(_yaml_scalar "$CONFIG_FILE" testing opening_plies)"
_cfg_seed="$(_yaml_scalar "$CONFIG_FILE" testing opening_seed)"
_cfg_depth="$(_yaml_scalar "$CONFIG_FILE" testing inference_depth)"
_cfg_max_moves="$(_yaml_scalar "$CONFIG_FILE" selfplay max_moves_per_game)"

NUM_GAMES="${NUM_GAMES:-${_cfg_games:-100}}"
DIFFICULTIES="${DIFFICULTIES:-${_cfg_difficulty:-easy}}"
MAX_MOVES="${MAX_MOVES:-${_cfg_max_moves:-200}}"
OPENING_PLIES="${OPENING_PLIES:-${_cfg_plies:-0,2,4,6,8}}"
OPENING_SEED="${OPENING_SEED:-${_cfg_seed:-20260819}}"
INFERENCE_DEPTH="${INFERENCE_DEPTH:-${_cfg_depth:-1}}"

# A flow-style YAML list ("[2, 4, 6, 8]") becomes "2,4,6,8".
OPENING_PLIES="$(printf '%s' "$OPENING_PLIES" | tr -d '[] ')"

_is_uint() { [[ "$1" =~ ^[0-9]{1,18}$ ]]; }
_positive_int() {
    # _positive_int <name> <value>: prints the value in base 10 ("08" -> 8).
    _is_uint "$2" && (( 10#$2 > 0 )) || die "$1 must be a positive integer (got: '$2')"
    printf '%d' "$(( 10#$2 ))"
}
NUM_GAMES="$(_positive_int NUM_GAMES "$NUM_GAMES")"
(( NUM_GAMES % 2 == 0 )) || die "NUM_GAMES must be even so each side plays exactly half (got: $NUM_GAMES)"
NUM_WORKERS="$(_positive_int NUM_WORKERS "$NUM_WORKERS")"
MAX_MOVES="$(_positive_int MAX_MOVES "$MAX_MOVES")"
[[ "$OPENING_PLIES" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "OPENING_PLIES must be a comma list of integers (got: '$OPENING_PLIES')"
_is_uint "$OPENING_SEED" || die "OPENING_SEED must be a non-negative integer (got: '$OPENING_SEED')"
OPENING_SEED="$(( 10#$OPENING_SEED ))"
[[ "$INFERENCE_DEPTH" =~ ^[123]$ ]] || die "INFERENCE_DEPTH must be 1, 2, or 3 (got: '$INFERENCE_DEPTH')"
if [[ -n "$MIN_WIN_RATE" ]]; then
    [[ "$MIN_WIN_RATE" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] \
        && awk -v v="$MIN_WIN_RATE" 'BEGIN { exit !(v <= 100) }' \
        || die "MIN_WIN_RATE must be a percentage from 0 to 100 (got: '$MIN_WIN_RATE')"
fi

# Unknown names silently get the medium budget inside get_best_move(), so
# reject them here instead of mislabelling a medium run.
_requested=()
for _d in ${DIFFICULTIES//,/ }; do
    if [[ "$_d" == "all" ]]; then
        _requested+=("${ALL_DIFFICULTIES[@]}")
    else
        _requested+=("$_d")
    fi
done
DIFFICULTY_LIST=()
for _d in "${_requested[@]}"; do
    _known=false
    for _k in "${ALL_DIFFICULTIES[@]}"; do
        [[ "$_d" == "$_k" ]] && _known=true
    done
    $_known || die "Unknown difficulty '$_d' in DIFFICULTIES (expected: ${ALL_DIFFICULTIES[*]} or all)"
    _dup=false
    for _k in "${DIFFICULTY_LIST[@]}"; do
        [[ "$_d" == "$_k" ]] && _dup=true
    done
    $_dup || DIFFICULTY_LIST+=("$_d")
done
(( ${#DIFFICULTY_LIST[@]} > 0 )) || die "DIFFICULTIES selects no difficulty"

if [[ -z "$STATS_DIR" ]]; then
    if [[ -n "$OUTPUT_NAMESPACE" ]]; then
        STATS_DIR="models/test_stats/$OUTPUT_NAMESPACE/win_rate_tester"
    else
        STATS_DIR="models/test_stats/win_rate_tester"
    fi
fi
STATS_DIR="$(_abs_project_path "$STATS_DIR")"
RESULTS_FILE="$(_abs_project_path "${RESULTS_FILE:-$STATS_DIR/results.jsonl}")"

# plot_training.py finds these runs by globbing results.jsonl under the
# test_stats directory beside the config's stats file (_resolve_test_stats_dir
# and load_test_stats_results). A results file under another name, or outside
# that tree, is measured correctly and then never reaches the dashboard, so say
# so here rather than letting the panel come up silently empty.
_cfg_stats_file="$(_yaml_scalar "$CONFIG_FILE" paths stats_file)"
DASHBOARD_TEST_STATS_DIR="$(_abs_project_path \
    "$(dirname "${_cfg_stats_file:-models/training_stats.json}")/test_stats")"
if [[ "$(basename "$RESULTS_FILE")" != "results.jsonl" ]]; then
    log "WARN: RESULTS_FILE is not named results.jsonl, so plot_training.py will not plot this run."
elif [[ "$RESULTS_FILE" != "$DASHBOARD_TEST_STATS_DIR"/* ]]; then
    log "WARN: RESULTS_FILE is outside $(_rel_project_path "$DASHBOARD_TEST_STATS_DIR"), so plot_training.py will not plot this run."
fi

# ========================== INTERPRETER ======================================

# Never let this fall back to conda base: the Cython extensions are CPython
# 3.11 and a base interpreter silently plays the ~100x slower Python search.
PYTHON_CMD=()
if [[ -n "${DAMA_PYTHON:-}" ]]; then
    [[ -x "$DAMA_PYTHON" ]] || die "DAMA_PYTHON is not executable: $DAMA_PYTHON"
    PYTHON_CMD=("$DAMA_PYTHON")
elif [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]] && command -v python >/dev/null 2>&1; then
    PYTHON_CMD=("$(command -v python)")
else
    _conda_bin="${CONDA_EXE:-}"
    [[ -x "$_conda_bin" ]] || _conda_bin="$(command -v conda 2>/dev/null || true)"
    if [[ ! -x "$_conda_bin" ]]; then
        for _candidate in "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" \
                          "$HOME/miniforge3/bin/conda" "$HOME/mambaforge/bin/conda" \
                          "/opt/conda/bin/conda"; do
            if [[ -x "$_candidate" ]]; then
                _conda_bin="$_candidate"
                break
            fi
        done
    fi
    [[ -x "$_conda_bin" ]] || die "The '$CONDA_ENV' interpreter is required. Run bash setup_conda.sh."
    if ! _dama_python="$("$_conda_bin" run -n "$CONDA_ENV" python -c \
        'import sys; print(sys.executable)' 2>/dev/null)" || [[ ! -x "$_dama_python" ]]; then
        die "Could not resolve the '$CONDA_ENV' interpreter. Run bash setup_conda.sh."
    fi
    PYTHON_CMD=("$_dama_python")
fi

export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
if ! _python_info="$("${PYTHON_CMD[@]}" -c '
import sys
from dama.ai.algorithmic.search import _HAS_FAST_SEARCH
if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"expected CPython 3.11, got {sys.version.split()[0]}")
if not _HAS_FAST_SEARCH:
    raise SystemExit("the compiled fast-search extension did not load")
print(sys.executable)
' 2>&1)"; then
    die "The dama interpreter preflight failed: $_python_info"
fi

# ========================== PIN THE MODEL ====================================

# Every game worker loads the model path on its own, and the trainer atomically
# replaces the latest/promoted aliases and prunes numbered checkpoints while it
# runs. A private hard link (or a copy across filesystems) holds one inode for
# the whole run, so every game plays the same checkpoint revision.
mkdir -p "$STATS_DIR" "$(dirname "$RESULTS_FILE")"
# A killed run cannot clean up; drop pins whose owning shell is gone, since a
# pin can hold a pruned checkpoint's space.
for _stale in "$STATS_DIR"/.pinned_*; do
    [[ -d "$_stale" ]] || continue
    _owner="${_stale##*_}"
    if [[ "$_owner" =~ ^[0-9]+$ ]] && ! kill -0 "$_owner" 2>/dev/null; then
        rm -rf -- "$_stale"
    fi
done
PIN_DIR="$STATS_DIR/.pinned_$$"
RECORDS_FILE=""
_cleanup() {
    rm -rf -- "$PIN_DIR"
    [[ -z "$RECORDS_FILE" ]] || rm -f -- "$RECORDS_FILE"
}
trap _cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir -p "$PIN_DIR"

_pin_model() {
    # _pin_model <model file>: prints the pinned path.
    local source="$1"
    local pinned
    pinned="$PIN_DIR/$(basename "$source")"
    if ! ln -- "$source" "$pinned" 2>/dev/null; then
        cp -- "$source" "$pinned" || die "Could not pin a private copy of $source"
    fi
    printf '%s' "$pinned"
}

_model_metadata() {
    # _model_metadata <pinned file>: prints "<step> <sha256>", step -1 when unknown.
    # A numbered checkpoint's step comes from its name, so a sweep of many
    # models does not pay a torch import and archive open per model.
    # stderr stays on the terminal so torch warnings cannot corrupt the line.
    PINNED_MODEL="$1" "${PYTHON_CMD[@]}" -c '
import hashlib, os, re
path = os.environ["PINNED_MODEL"]
digest = hashlib.sha256()
with open(path, "rb") as handle:
    for block in iter(lambda: handle.read(1 << 20), b""):
        digest.update(block)
match = re.search(r"model_step_0*(\d+)\.pt$", path)
step = int(match.group(1)) if match else None
if step is None:
    try:
        import torch
        checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("step"), int):
            step = checkpoint["step"]
    except Exception:
        pass
print(-1 if step is None else step, digest.hexdigest().upper())
'
}

# Resume support: collect the (checkpoint sha256, difficulty) pairs already
# measured at these exact settings, so an interrupted sweep does not repeat
# finished work. Keyed on the checkpoint's own bytes rather than its step,
# because dead-epoch rollback can republish a step number with different
# weights, and that replacement must be measured again.
declare -A ALREADY_TESTED=()
if [[ "$SKIP_ALREADY_TESTED" == true && -f "$RESULTS_FILE" ]]; then
    while IFS= read -r _key; do
        [[ -n "$_key" ]] && ALREADY_TESTED["$_key"]=1
    done < <(
        RESULTS_FILE="$RESULTS_FILE" EXPECT_GAMES="$NUM_GAMES" \
        EXPECT_SEED="$OPENING_SEED" EXPECT_PLIES="$OPENING_PLIES" \
        EXPECT_DEPTH="$INFERENCE_DEPTH" EXPECT_MAX_MOVES="$MAX_MOVES" \
            "${PYTHON_CMD[@]}" -c '
import json, os
expect_games = int(os.environ["EXPECT_GAMES"])
expect_seed = int(os.environ["EXPECT_SEED"])
expect_depth = int(os.environ["EXPECT_DEPTH"])
expect_max_moves = int(os.environ["EXPECT_MAX_MOVES"])
expect_plies = [int(p) for p in os.environ["EXPECT_PLIES"].split(",") if p.strip()]
with open(os.environ["RESULTS_FILE"], "r", encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("total_games") != expect_games:
            continue
        if record.get("opening_seed") != expect_seed:
            continue
        if list(record.get("opening_plies") or []) != expect_plies:
            continue
        if record.get("ml_inference_depth") != expect_depth:
            continue
        if record.get("max_moves") != expect_max_moves:
            continue
        sha256, difficulty = record.get("model_sha256"), record.get("algo_difficulty")
        if sha256 and difficulty:
            print(f"{sha256}|{difficulty}")
' 2>/dev/null
    )
fi

# ========================== EVALUATE =========================================

RUN_ID="$(date -u '+%Y%m%dT%H%M%SZ')_$$"
RECORDS_FILE="$(mktemp "${TMPDIR:-/tmp}/ml_vs_algo_records.XXXXXX")"

log "Config:   $(_rel_project_path "$CONFIG_FILE") (namespace: ${OUTPUT_NAMESPACE:-legacy})"
log "Python:   $_python_info"
if (( ${#MODEL_FILES[@]} == 1 )); then
    log "Model:    $(_rel_project_path "${MODEL_FILES[0]}")"
else
    log "Models:   ${#MODEL_FILES[@]} checkpoints, $(_rel_project_path "$(dirname "${MODEL_FILES[0]}")")"
    log "          $(basename "${MODEL_FILES[0]}") .. $(basename "${MODEL_FILES[-1]}")"
fi
log "Settings: ${#DIFFICULTY_LIST[@]} difficulty(ies) [${DIFFICULTY_LIST[*]}], $NUM_GAMES games each, $NUM_WORKERS workers, max $MAX_MOVES moves"
log "Openings: plies $OPENING_PLIES, seed $OPENING_SEED, inference depth $INFERENCE_DEPTH"

# Game workers share the CPU with a live trainer. The pid in run_status.json is
# authoritative; `pgrep -x micro-trainer` misses a trainer whose title carries
# the live step suffix.
_log_dir="$(_yaml_scalar "$CONFIG_FILE" paths log_dir)"
_status_file="$(_abs_project_path "${_log_dir:-logs}")/run_status.json"
if [[ -f "$_status_file" ]] && grep -q '"status": *"running"' "$_status_file"; then
    _trainer_pid="$(sed -n 's/.*"pid": *\([0-9][0-9]*\).*/\1/p' "$_status_file" | head -n 1)"
    if [[ -n "$_trainer_pid" && -r "/proc/$_trainer_pid/cmdline" ]] \
       && tr '\0' ' ' < "/proc/$_trainer_pid/cmdline" | grep -Eq 'dama\.ai\.ml\.trainer|micro[-_]trainer'; then
        log "WARN: a trainer is running (pid $_trainer_pid); games share its CPU and run slower."
        for _d in "${DIFFICULTY_LIST[@]}"; do
            if [[ "$_d" != "easy" ]]; then
                log "WARN: '$_d' search is time-budgeted, so CPU contention can make the algorithm weaker than usual."
                break
            fi
        done
    fi
fi

MODEL_INDEX=0
EVALUATED=0
SKIPPED=0
for MODEL_FILE in "${MODEL_FILES[@]}"; do
    MODEL_INDEX=$(( MODEL_INDEX + 1 ))
    _model_name="$(basename "$MODEL_FILE")"
    _progress=""
    (( ${#MODEL_FILES[@]} > 1 )) && _progress="[$MODEL_INDEX/${#MODEL_FILES[@]}] "

    PINNED_MODEL="$(_pin_model "$MODEL_FILE")"
    if ! _model_info="$(_model_metadata "$PINNED_MODEL")"; then
        die "Could not read model metadata from $MODEL_FILE"
    fi
    read -r MODEL_STEP MODEL_SHA256 <<<"$(printf '%s\n' "$_model_info" | tail -n 1)"
    if ! _is_uint "$MODEL_STEP"; then
        log "WARN: no training step recorded in $_model_name; reporting step as unknown."
        MODEL_STEP=0
        STEP_KNOWN=false
    else
        STEP_KNOWN=true
    fi
    log "MODEL: ${_progress}$_model_name (step $($STEP_KNOWN && echo "$MODEL_STEP" || echo unknown), sha256 ${MODEL_SHA256:0:16}...)"

    for difficulty in "${DIFFICULTY_LIST[@]}"; do
        if [[ -n "${ALREADY_TESTED["$MODEL_SHA256|$difficulty"]:-}" ]]; then
            log "SKIP: $_model_name vs $difficulty already measured at these settings"
            SKIPPED=$(( SKIPPED + 1 ))
            continue
        fi
        log "EVAL: ${_progress}vs $difficulty algorithm - $NUM_GAMES games"
        _started=$SECONDS
        # A real module, never `python -`: forkserver/spawn workers re-import the
        # main module. stdout carries only the final JSON record.
        if ! json_line="$(
            cd "$PROJECT_DIR"
            "${PYTHON_CMD[@]}" -m dama.ai.ml.eval_checkpoint_once \
                --checkpoint "$PINNED_MODEL" \
                --checkpoint-name "$_model_name" \
                --step "$MODEL_STEP" \
                --num-games "$NUM_GAMES" \
                --difficulty "$difficulty" \
                --opponent algorithm \
                --num-workers "$NUM_WORKERS" \
                --max-moves "$MAX_MOVES" \
                --stats-dir "$STATS_DIR" \
                --opening-plies "$OPENING_PLIES" \
                --opening-seed "$OPENING_SEED" \
                --inference-depth "$INFERENCE_DEPTH"
        )"; then
            log "ERROR: evaluation of $_model_name vs $difficulty failed"
            exit 2
        fi
        # Tab-separated provenance per record: a sweep tests many models, and
        # compact JSON never contains a literal tab.
        printf '%s\t%s\t%s\t%s\n' \
            "$MODEL_FILE" "$MODEL_SHA256" "$STEP_KNOWN" \
            "$(printf '%s\n' "$json_line" | tail -n 1)" >> "$RECORDS_FILE"
        EVALUATED=$(( EVALUATED + 1 ))
        log "DONE: vs $difficulty in $(( SECONDS - _started ))s"
    done

    # Release the pin before the next model so a sweep holds at most one
    # checkpoint that its source directory may have already pruned.
    rm -f -- "$PINNED_MODEL"
done

if (( EVALUATED == 0 )); then
    log "Nothing to evaluate: $SKIPPED measurement(s) already recorded at these settings."
    log "Set SKIP_ALREADY_TESTED=false to measure them again."
    exit 0
fi
(( SKIPPED == 0 )) || log "Skipped $SKIPPED already-measured model/difficulty pair(s)."

# ========================== REPORT ===========================================

set +e
RECORDS_FILE="$RECORDS_FILE" RESULTS_FILE="$RESULTS_FILE" RUN_ID="$RUN_ID" \
CONFIG_FILE="$CONFIG_FILE" PROJECT_DIR="$PROJECT_DIR" OUTPUT_NAMESPACE="$OUTPUT_NAMESPACE" \
MAX_MOVES="$MAX_MOVES" MIN_WIN_RATE="$MIN_WIN_RATE" STATS_DIR="$STATS_DIR" \
OVERWRITE_PREVIOUS="$OVERWRITE_PREVIOUS" \
    "${PYTHON_CMD[@]}" - <<'PY'
import json
import os
import sys
import tempfile

from dama.ai.ml.acceptance import wdl_summary

env = os.environ
project_dir = env["PROJECT_DIR"]


def rel(path):
    prefix = project_dir.rstrip("/") + "/"
    return path[len(prefix):] if path.startswith(prefix) else path


shared_provenance = {
    "tester": "test_ml_vs_algo_win_rate.sh",
    "run_id": env["RUN_ID"],
    "config": rel(env["CONFIG_FILE"]),
    "namespace": env["OUTPUT_NAMESPACE"] or None,
    "max_moves": int(env["MAX_MOVES"]),
}

records = []
with open(env["RECORDS_FILE"], "r", encoding="utf-8") as handle:
    for line in handle:
        line = line.rstrip("\n")
        if not line:
            continue
        # <model file>\t<sha256>\t<step known>\t<evaluation JSON>
        parts = line.split("\t", 3)
        if len(parts) != 4:
            print(f"ERROR: malformed evaluation line: {line[:200]}", file=sys.stderr)
            sys.exit(2)
        model_file, model_sha256, step_known_text, payload = parts
        try:
            record = json.loads(payload)
        except json.JSONDecodeError:
            print(f"ERROR: invalid evaluation record: {payload[:200]}", file=sys.stderr)
            sys.exit(2)
        record.update(shared_provenance)
        record["model"] = rel(model_file)
        record["model_sha256"] = model_sha256
        record["step_known"] = step_known_text == "true"
        records.append(record)
if not records:
    print("ERROR: no evaluation records were produced", file=sys.stderr)
    sys.exit(2)

def result_key(record):
    """What counts as the same measurement point for overwriting.

    The dashboard plots one point per (step, difficulty), so that is what a
    re-test replaces. A checkpoint with no resolved step is never plotted and
    is identified by its own bytes instead.
    """
    difficulty = record.get("algo_difficulty")
    if record.get("step_known"):
        return ("step", record.get("step"), difficulty)
    return ("sha256", record.get("model_sha256"), difficulty)


results_path = env["RESULTS_FILE"]
superseded = 0
if env["OVERWRITE_PREVIOUS"] == "true" and os.path.exists(results_path):
    replaced_keys = {result_key(record) for record in records}
    kept = []
    with open(results_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                existing = json.loads(line)
            except json.JSONDecodeError:
                kept.append(line)  # never discard what cannot be understood
                continue
            if result_key(existing) in replaced_keys:
                superseded += 1
                continue
            kept.append(line)

    # Rewrite through a temporary in the same directory: a partially written
    # results file would lose measurements this run did not take.
    directory = os.path.dirname(results_path) or "."
    descriptor, temporary = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for line in kept:
                handle.write(line + "\n")
            for record in records:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, results_path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
else:
    with open(results_path, "a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")

models = {}
for record in records:
    models.setdefault(record["model_sha256"], record)
single_model = len(models) == 1


def pct(value):
    return f"{value * 100:6.1f}%"


def side_rate(wdl):
    return wdl["wins"] / wdl["total"] if wdl["total"] else 0.0


width = 96 if single_model else 107
rule = "=" * width
thin = "-" * width
step_column = "" if single_model else f"{'Step':>10}  "
head = (f" {step_column}{'Opponent':<12}{'Games':>6}{'W':>6}{'D':>5}{'L':>6}"
        f"{'Win rate':>10}{'As P1':>9}{'As P2':>9}   {'Match score [Wilson 95%]':<27}{'Avg len':>7}")

first = records[0]
print()
print(rule)
print(" ML Model vs Algorithm - Overall Win Rate")
print(rule)
if single_model:
    step_text = f"{first['step']:,}" if first["step_known"] else "unknown"
    print(f" Model     {first['model']}")
    print(f" Step      {step_text}    sha256 {first['model_sha256'][:16]}...")
else:
    steps = sorted(record["step"] for record in records if record["step_known"])
    span = f"steps {steps[0]:,} to {steps[-1]:,}" if steps else "steps unknown"
    print(f" Models    {len(models)} checkpoints ({span})")
print(f" Config    {first['config']} (namespace: {first['namespace'] or 'legacy'})")
print(f" Openings  plies {','.join(str(p) for p in first['opening_plies'])}, "
      f"seed {first['opening_seed']}, suite {first['opening_suite_id'][:19]}...")
print(f" Rules     {first['total_games']} games per opponent, half per side, "
      f"max {first['max_moves']} moves, inference depth {first['ml_inference_depth']}")
print(thin)
print(head)
print(thin)

totals = {"wins": 0, "draws": 0, "losses": 0}
p1 = {"wins": 0, "draws": 0, "losses": 0, "total": 0}
p2 = {"wins": 0, "draws": 0, "losses": 0, "total": 0}
moves = 0.0
for record in sorted(records, key=lambda item: (item["step"], item["algo_difficulty"])):
    overall = record["overall_wdl"]
    ci = record["match_score_ci_95"]
    for key in totals:
        totals[key] += overall[key]
    for bucket, wdl in ((p1, record["ml_as_p1_wdl"]), (p2, record["ml_as_p2_wdl"])):
        for key in bucket:
            bucket[key] += wdl[key]
    moves += record["avg_game_length"] * record["total_games"]
    score = f"{record['match_score']:.3f} [{ci['lower']:.3f}, {ci['upper']:.3f}]"
    step_cell = "" if single_model else (
        f"{record['step']:>10,}  " if record["step_known"] else f"{'unknown':>10}  ")
    print(f" {step_cell}{record['algo_difficulty']:<12}{record['total_games']:>6}{overall['wins']:>6}"
          f"{overall['draws']:>5}{overall['losses']:>6}{pct(record['ml_win_rate']):>10}"
          f"{pct(record['ml_as_p1_win_rate']):>9}{pct(record['ml_as_p2_win_rate']):>9}"
          f"   {score:<27}{record['avg_game_length']:>7.1f}")

games = sum(totals.values())
overall_rate = totals["wins"] / games
pooled = wdl_summary(totals["wins"], totals["draws"], totals["losses"])
if len(records) == 1:
    ci = pooled["match_score_ci_95"]
    pooled_score = f"{pooled['match_score']:.3f} [{ci['lower']:.3f}, {ci['upper']:.3f}]"
else:
    # One interval over a mixture of models and opponents is not a meaningful
    # estimate of any single matchup.
    pooled_score = f"{pooled['match_score']:.3f} (pooled, no CI)"
print(thin)
overall_step_cell = "" if single_model else f"{'':>10}  "
print(f" {overall_step_cell}{'Overall':<12}{games:>6}{totals['wins']:>6}{totals['draws']:>5}{totals['losses']:>6}"
      f"{pct(overall_rate):>10}{pct(side_rate(p1)):>9}{pct(side_rate(p2)):>9}"
      f"   {pooled_score:<27}{moves / games:>7.1f}")
print(rule)
if superseded:
    print(f" Results   {rel(env['RESULTS_FILE'])}  ({superseded} earlier record(s) replaced)")
else:
    print(f" Results   {rel(env['RESULTS_FILE'])}")
print(f" Details   {rel(env['STATS_DIR'])}/test_*.json")

exit_code = 0
if env["MIN_WIN_RATE"]:
    threshold = float(env["MIN_WIN_RATE"])
    passed = overall_rate * 100 >= threshold
    verdict = "PASS" if passed else "FAIL"
    print(f" Gate      {verdict}: overall win rate {overall_rate * 100:.1f}% "
          f"{'>=' if passed else '<'} {threshold:g}%")
    exit_code = 0 if passed else 3
print(rule)
sys.exit(exit_code)
PY
_status=$?
set -e
exit "$_status"
