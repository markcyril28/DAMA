#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# LOCAL TRAINING LAUNCHER
# =============================================================================
# LOCAL SPECS:
#   GPU: NVIDIA RTX 5050 (8GB VRAM)
#   RAM: 24 GB visible to WSL
#   CPU Cores: 12
#
# All training parameters are in the selected config below.
# This script only handles session-level overrides (resume, duration, etc.).
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIG SELECTION — Uncomment exactly one
# -----------------------------------------------------------------------------
#TRAINING_CONFIG="config/training_config_local_retrain.yaml"
# TRAINING_CONFIG="config/training_config_local.yaml"
# TRAINING_CONFIG="config/training_config.yaml"
# TRAINING_CONFIG="config/training_config_server.yaml"
# TRAINING_CONFIG="config/training_config_server_retrain.yaml"
# Retired: config/superseded/training_config_policy_distillation.yaml (step 134000,
# writes into the preserved wd1e4 namespace). Do not point TRAINING_CONFIG at it.
TRAINING_CONFIG="config/training_config_policy_distillation_c174k.yaml"

# -----------------------------------------------------------------------------
# SESSION SETTINGS — Edit these per-run
# -----------------------------------------------------------------------------
CONDA_ENV="dama"                 # Conda env auto-activated if not already active (see setup_conda.sh)
SET_PROCESS_TITLE=true           # Set to false to disable custom process title in htop
PROCESS_TITLE="micro-trainer"            # Process name shown in htop (requires 'setproctitle' package)
RESUME_LATEST="auto"          # auto = policy baseline, latest for other configs; true/false remain supported.
RESUME=""                     # Optional checkpoint. Enhanced stage requires the recorded promoted checkpoint.
RESUME_CONTINUATION=true      # Recovery only: a relaunch continues this namespace's newest lineage-verified
                              # checkpoint instead of re-walking from the anchor (audit Suggestion 7).
                              # false restores anchor-only restarts. Ignored unless the config pins a recovery baseline.
ENHANCED_STAGE=false          # true only after policy-only promotion; trainer verifies registry, hash, and suite.
ENHANCED_INFERENCE_DEPTH=2    # Enhanced stage only: supported values are 2 or 3.
TRAIN_DURATION=""                # Train for this duration (empty = use config's time_limit)
                                 # Examples: "2d", "4h", "30m", "1d12h"
MIN_FREE_DISK_GB=10              # Refuse to launch below this much free space on the project
                                 # volume (checkpoints + snapshots grow several GB/day). 0 disables.

# =============================================================================
# END OF PARAMETERS - Do not edit below this line
# =============================================================================

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

# Resolve the config before creating logs or invoking the trainer. The policy
# recovery is resume-only and tied to one preserved checkpoint. Other configs
# retain their normal latest, specific-checkpoint, and fresh-start modes.
if [[ "$TRAINING_CONFIG" = /* ]]; then
    SELECTED_CONFIG_PATH="$TRAINING_CONFIG"
else
    SELECTED_CONFIG_PATH="${PROJECT_DIR}/${TRAINING_CONFIG}"
fi
if [ ! -f "$SELECTED_CONFIG_PATH" ]; then
    echo "ERROR: Training config not found: $SELECTED_CONFIG_PATH" >&2
    exit 1
fi
SELECTED_CONFIG_PATH="$(readlink -f "$SELECTED_CONFIG_PATH")"

# The recovery anchor used to be hardcoded here as step 134000. The approved
# 2026-08-24 continuation resumes step 174000 in a new namespace, so a second
# hardcoded copy would simply be a second thing to forget to update. Read the
# anchor, its pinned digest, and the output stats path out of the selected
# config instead -- the trainer validates the same fields, so the two can no
# longer disagree.
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
_abs_project_path() { case "$1" in /*) printf '%s' "$1" ;; *) printf '%s/%s' "$PROJECT_DIR" "$1" ;; esac; }

POLICY_RECOVERY_ENABLED="$(_yaml_scalar "$SELECTED_CONFIG_PATH" recovery_experiment enabled)"
POLICY_RECOVERY_BASELINE=""
POLICY_RECOVERY_SHA256=""
POLICY_RECOVERY_LEGACY_STATS=""
POLICY_RECOVERY_STATS=""
# PyYAML resolves True/yes/on (any case) as booleans, and the trainer reads
# this same key through yaml.safe_load -- so the launcher must accept the same
# spellings or it would silently skip every recovery guard below on a config
# that says `enabled: Yes`.
IS_POLICY_RECOVERY=false
case "$POLICY_RECOVERY_ENABLED" in
    true|True|TRUE|yes|Yes|YES|on|On|ON)
        IS_POLICY_RECOVERY=true
        ;;
    false|False|FALSE|no|No|NO|off|Off|OFF|"") ;;
    *)
        echo "ERROR: recovery_experiment.enabled is not a YAML boolean: '$POLICY_RECOVERY_ENABLED'" >&2
        exit 1
        ;;
esac
if [ "$IS_POLICY_RECOVERY" = true ]; then
    _baseline_rel="$(_yaml_scalar "$SELECTED_CONFIG_PATH" recovery_experiment baseline_checkpoint)"
    POLICY_RECOVERY_SHA256="$(_yaml_scalar "$SELECTED_CONFIG_PATH" recovery_experiment baseline_sha256 | tr 'a-f' 'A-F')"
    _stats_rel="$(_yaml_scalar "$SELECTED_CONFIG_PATH" paths stats_file)"
    _seed_stats_rel="$(_yaml_scalar "$SELECTED_CONFIG_PATH" paths seed_stats_from)"
    if [ -z "$_baseline_rel" ] || [ -z "$POLICY_RECOVERY_SHA256" ]; then
        echo "ERROR: recovery_experiment needs baseline_checkpoint and baseline_sha256 in $SELECTED_CONFIG_PATH" >&2
        exit 1
    fi
    POLICY_RECOVERY_BASELINE="$(_abs_project_path "$_baseline_rel")"
    [ -n "$_stats_rel" ] && POLICY_RECOVERY_STATS="$(_abs_project_path "$_stats_rel")"
    [ -n "$_seed_stats_rel" ] && POLICY_RECOVERY_LEGACY_STATS="$(_abs_project_path "$_seed_stats_rel")"
fi

case "$RESUME_LATEST" in
    auto|true|false) ;;
    *)
        echo "ERROR: RESUME_LATEST must be auto, true, or false." >&2
        exit 1
        ;;
esac
case "$ENHANCED_STAGE" in
    true|false) ;;
    *)
        echo "ERROR: ENHANCED_STAGE must be true or false." >&2
        exit 1
        ;;
esac
case "$RESUME_CONTINUATION" in
    true|false) ;;
    *)
        echo "ERROR: RESUME_CONTINUATION must be true or false." >&2
        exit 1
        ;;
esac

case "$MIN_FREE_DISK_GB" in
    ''|*[!0-9]*)
        echo "ERROR: MIN_FREE_DISK_GB must be a non-negative integer (GB)." >&2
        exit 1
        ;;
esac

# Approximation of the trainer's parse_duration() grammar, so a mistyped
# duration fails in preflight instead of as a traceback after conda setup,
# the Cython scan, and CUDA init. parse_duration() is deliberately lenient:
# a bare number is hours, a unit word anywhere after digits counts ("2d4x",
# "1.5d", "-4h" all parse), and only strings with no digit+unit pair and no
# numeric reading are rejected. This gate mirrors exactly that accept set;
# the only divergence is "0h", which the trainer rejects for being zero but
# this gate accepts (the trainer raises on it at startup either way).
if [ -n "$TRAIN_DURATION" ]; then
    if ! printf '%s' "$TRAIN_DURATION" | \
         grep -Eq '[0-9]+[[:space:]]*(d(ays?)?|h(ours?)?|m(in(utes?)?)?|s(ec(onds?)?)?)' \
         && ! [[ "$TRAIN_DURATION" =~ ^[0-9]+([.][0-9]+)?$ ]] \
         && ! [[ "$TRAIN_DURATION" =~ ^[.][0-9]+$ ]] \
         && ! [[ "$TRAIN_DURATION" =~ ^[0-9]+[.]$ ]]; then
        echo "ERROR: TRAIN_DURATION '$TRAIN_DURATION' is not parseable. Use e.g. \"2d\", \"4h\", \"30m\", \"1d12h\"." >&2
        exit 1
    fi
fi

# Local runs write several GB/day of checkpoints and corpus snapshots (the
# audited 48h run exhausted its volume near the 24h mark), so refuse to launch
# below the floor instead of dying mid-run with a partial checkpoint set.
if [ "$MIN_FREE_DISK_GB" -gt 0 ]; then
    _free_kb="$(df -Pk -- "$PROJECT_DIR" 2>/dev/null | awk 'NR==2 {print $4}')" || _free_kb=""
    if [ -n "$_free_kb" ] && [ "$_free_kb" -lt $((MIN_FREE_DISK_GB * 1024 * 1024)) ]; then
        echo "ERROR: Only $(( _free_kb / 1024 / 1024 )) GB free on ${PROJECT_DIR}'s volume;" >&2
        echo "       MIN_FREE_DISK_GB=${MIN_FREE_DISK_GB} required before launching training." >&2
        echo "       Free up space, or lower the floor if this run writes less than usual." >&2
        exit 1
    elif [ -z "$_free_kb" ]; then
        echo "[warn] Could not measure free disk space; skipping the MIN_FREE_DISK_GB gate."
    fi
fi
if [ "$ENHANCED_STAGE" = true ] && [ "$IS_POLICY_RECOVERY" = false ]; then
    echo "ERROR: ENHANCED_STAGE is available only with a recovery_experiment config." >&2
    exit 1
fi
if [ "$ENHANCED_STAGE" = true ] &&
   [ "$ENHANCED_INFERENCE_DEPTH" != 2 ] &&
   [ "$ENHANCED_INFERENCE_DEPTH" != 3 ]; then
    echo "ERROR: ENHANCED_INFERENCE_DEPTH must be 2 or 3." >&2
    exit 1
fi

USE_RESUME_CONTINUATION=false

if [ "$IS_POLICY_RECOVERY" = true ]; then
    if [ "$RESUME_LATEST" = true ]; then
        echo "ERROR: --resume-latest is disabled for the policy-distillation recovery experiment." >&2
        exit 1
    fi
    if [ ! -f "$POLICY_RECOVERY_BASELINE" ]; then
        echo "ERROR: Policy recovery baseline not found: $POLICY_RECOVERY_BASELINE" >&2
        exit 1
    fi

    if [ "$ENHANCED_STAGE" = true ]; then
        if [ -z "$RESUME" ]; then
            echo "ERROR: ENHANCED_STAGE requires the recorded promoted policy-only checkpoint in RESUME." >&2
            exit 1
        fi
        if [[ "$RESUME" = /* ]]; then
            _resume_candidate="$RESUME"
        else
            _resume_candidate="${PROJECT_DIR}/${RESUME}"
        fi
        if [ ! -f "$_resume_candidate" ]; then
            echo "ERROR: Promoted policy checkpoint not found: $_resume_candidate" >&2
            exit 1
        fi
        RESUME="$(readlink -f "$_resume_candidate")"
    else
        if [ -z "$RESUME" ]; then
            if [ "$RESUME_LATEST" != auto ]; then
                echo "ERROR: Fresh start is disabled for policy recovery. Resume from ${POLICY_RECOVERY_BASELINE}." >&2
                exit 1
            fi
            RESUME="$POLICY_RECOVERY_BASELINE"
            # Audit Suggestion 7. Only the implicit path opts in: an operator
            # who names a checkpoint explicitly gets exactly that checkpoint.
            # The trainer re-verifies the lineage stamp and falls back to this
            # anchor when the namespace holds no verified checkpoint yet, so
            # the first launch of a namespace is unaffected.
            [ "$RESUME_CONTINUATION" = true ] && USE_RESUME_CONTINUATION=true
        else
            if [[ "$RESUME" = /* ]]; then
                _resume_candidate="$RESUME"
            else
                _resume_candidate="${PROJECT_DIR}/${RESUME}"
            fi
            if [ ! -f "$_resume_candidate" ]; then
                echo "ERROR: Recovery checkpoint not found: $_resume_candidate" >&2
                exit 1
            fi
            _resume_candidate="$(readlink -f "$_resume_candidate")"
            if [ "$_resume_candidate" != "$(readlink -f "$POLICY_RECOVERY_BASELINE")" ]; then
                echo "ERROR: Policy recovery must resume from: $POLICY_RECOVERY_BASELINE" >&2
                exit 1
            fi
            RESUME="$POLICY_RECOVERY_BASELINE"
        fi
    fi

    _actual_recovery_sha256="$(sha256sum "$POLICY_RECOVERY_BASELINE" | awk '{print toupper($1)}')"
    if [ "$_actual_recovery_sha256" != "$POLICY_RECOVERY_SHA256" ]; then
        echo "ERROR: Policy recovery baseline SHA-256 mismatch." >&2
        echo "Expected: $POLICY_RECOVERY_SHA256" >&2
        echo "Found:    $_actual_recovery_sha256" >&2
        echo "Training was not started." >&2
        exit 1
    fi
fi

# Every other config skipped this entirely: a mistyped RESUME reached the
# trainer unchecked and only failed after CUDA init, and a relative RESUME was
# forwarded verbatim -- so it resolved against the caller's working directory
# instead of the project root and broke when launched from anywhere else.
# Resolve it the same way the recovery branch already does.
if [ "$IS_POLICY_RECOVERY" = false ] && [ -n "$RESUME" ]; then
    if [[ "$RESUME" = /* ]]; then
        _resume_candidate="$RESUME"
    else
        _resume_candidate="${PROJECT_DIR}/${RESUME}"
    fi
    if [ ! -f "$_resume_candidate" ]; then
        echo "ERROR: Resume checkpoint not found: $_resume_candidate" >&2
        exit 1
    fi
    RESUME="$(readlink -f "$_resume_candidate")"
fi

# Add src to PYTHONPATH
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

# Process title shown in htop (stop_training.sh pattern-matches it). The false
# branch unsets rather than just skipping the export: the trainer reads
# PROCESS_TITLE from the environment, so a value inherited from the parent shell
# would otherwise survive and quietly ignore this toggle.
if [ "$SET_PROCESS_TITLE" = true ]; then export PROCESS_TITLE; else unset PROCESS_TITLE; fi

# Minimize CPU threads for numpy/MKL/OpenMP — self-play workers need the cores.
# The training thread uses GPU for all compute; these libraries would otherwise
# spawn threads that compete with the self-play worker processes
# (`selfplay.cpu_workers` in the config).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
# Trainer stdout goes through the tee pipe below, so Python block-buffers it.
# A hard kill (OOM, SIGHUP on terminal close) then discards the buffer and the
# console log ends at the last shell echo with no trainer output at all.
export PYTHONUNBUFFERED=1

# Console logging
LOG_DIR="${PROJECT_DIR}/logs/local/console"
mkdir -p "$LOG_DIR"
LOG_TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_DIR}/console_${LOG_TIMESTAMP}.txt"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Filipino Dama - ML Training (Local) ==="
echo ""
if [ "$IS_POLICY_RECOVERY" = true ]; then
    echo "Recovery baseline verified: ${POLICY_RECOVERY_BASELINE}"
    echo "Recovery SHA-256: ${POLICY_RECOVERY_SHA256}"
    if [ "$ENHANCED_STAGE" = false ] &&
       [ -n "$POLICY_RECOVERY_STATS" ] &&
       [ ! -e "$POLICY_RECOVERY_STATS" ] &&
       [ -n "$POLICY_RECOVERY_LEGACY_STATS" ] &&
       [ -f "$POLICY_RECOVERY_LEGACY_STATS" ]; then
        # Copy through a temp file and rename. A plain cp that is killed
        # part way leaves a truncated stats file behind, and both seeders
        # -- this one and the trainer's -- skip when the destination merely
        # exists, so the run would silently start from a corrupt history.
        _stats_seed_tmp="${POLICY_RECOVERY_STATS}.seed.tmp"
        if cp -- "$POLICY_RECOVERY_LEGACY_STATS" "$_stats_seed_tmp" &&
           mv -- "$_stats_seed_tmp" "$POLICY_RECOVERY_STATS"; then
            echo "Recovery stats seeded without modifying the legacy stats file: ${POLICY_RECOVERY_STATS}"
        else
            rm -f -- "$_stats_seed_tmp"
            echo "[warn] Could not seed recovery stats from ${POLICY_RECOVERY_LEGACY_STATS};"
            echo "[warn] the trainer will retry the same one-time copy."
        fi
    fi
    echo ""
fi

# -----------------------------------------------------------------------------
# Conda environment guard
# -----------------------------------------------------------------------------
# Everything below runs bare `python`, so the script must work when launched from
# `base`, from another env, or from a non-interactive shell with no env at all
# (cron / scripts/runner.sh, where .bashrc never runs and conda is not on PATH).
# Activate $CONDA_ENV when it isn't already active. Failures only warn: a working
# non-conda interpreter stays usable, and the CUDA check below is the real gate.
if [ "${CONDA_DEFAULT_ENV:-}" = "$CONDA_ENV" ]; then
    echo "Conda env: ${CONDA_ENV} (already active)"
else
    _prev_env="${CONDA_DEFAULT_ENV:-none}"

    # Locate the conda installation: exported CONDA_EXE, then PATH, then the
    # usual install prefixes (the only branch that works under cron).
    _conda_bin="${CONDA_EXE:-}"
    [ -x "$_conda_bin" ] || _conda_bin="$(command -v conda 2>/dev/null || true)"
    CONDA_BASE=""
    [ -x "$_conda_bin" ] && CONDA_BASE="$(dirname "$(dirname "$_conda_bin")")"
    # Guard on the empty case: with CONDA_BASE="" the test below reads
    # /etc/profile.d/conda.sh, which a system-wide conda install provides --
    # the fallback search would then be skipped and CONDA_BASE left empty.
    if [ -z "$CONDA_BASE" ] || [ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
        CONDA_BASE=""
        for _b in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" \
                  "$HOME/mambaforge" "$HOME/conda" "/opt/conda"; do
            if [ -f "$_b/etc/profile.d/conda.sh" ]; then CONDA_BASE="$_b"; break; fi
        done
    fi

    if [ -n "$CONDA_BASE" ]; then
        set +eu                      # conda.sh / activate are not -eu clean
        . "${CONDA_BASE}/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV"
        _activate_rc=$?
        set -euo pipefail
        if [ "$_activate_rc" -eq 0 ]; then
            echo "Conda env: ${CONDA_ENV} (activated; was '${_prev_env}')"
        else
            echo "[warn] Could not activate conda env '${CONDA_ENV}' (was '${_prev_env}')."
            echo "[warn] Create it with: bash setup_conda.sh; continuing with $(command -v python 2>/dev/null || echo python)"
        fi
    else
        echo "[warn] conda installation not found; continuing with $(command -v python 2>/dev/null || echo python)"
    fi
fi
echo ""

# Cython staleness guard (fail-safe). The .so files are NOT tracked (commit
# 2f85ac6 untracked them; .gitignore carries *.so), so a fresh clone has none
# at all and a local .pyx edit or a pull that changes a .pyx leaves the built
# one behind. Rebuild in-place when a .pyx is newer than its extension, or when
# the extension is missing. No-op when fresh; never blocks training on failure.
#
# The artifact is resolved by the running interpreter's EXT_SUFFIX rather than
# by `ls -t ... | head -1`: that glob matched every ABI tag and kept the newest
# by mtime, so a stray extension built for another Python version could mask a
# genuinely stale one. EXT_SUFFIX names exactly the file the *resolved*
# interpreter will import -- so this guard is only ABI-precise when `python`
# already IS the training env (the conda guard above normally guarantees it).
# If activation failed and `python` is another env, its ABI suffix makes the
# existing .so read as missing and every launch force-rebuilds with the wrong
# interpreter; the CUDA/import check below is what actually catches that.
_ext_suffix="$(python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX") or "")' 2>/dev/null || true)"
_cython_stale=false
while IFS= read -r _pyx; do
    if [ -n "$_ext_suffix" ]; then
        _so="${_pyx%.pyx}${_ext_suffix}"
        [ -f "$_so" ] || _so=""
    else
        # EXT_SUFFIX unavailable (no working interpreter yet): fall back to the
        # any-ABI glob rather than forcing a rebuild on every launch.
        _so="$(ls -t "${_pyx%.pyx}".*.so 2>/dev/null | head -1 || true)"
    fi
    if [ -z "$_so" ] || [ "$_pyx" -nt "$_so" ]; then _cython_stale=true; fi
done < <(find "${PROJECT_DIR}/src" -name '*.pyx' -not -path '*/build/*' 2>/dev/null)
if [ "$_cython_stale" = true ]; then
    echo "Cython sources changed — rebuilding extensions..."
    ( cd "${PROJECT_DIR}/src" && python setup_cython.py build_ext --inplace ) \
        || echo "[warn] Cython rebuild failed — using existing .so files."
    echo ""
fi

# Verify CUDA is available, and report whether the Cython accelerators loaded.
# An unbuilt or broken extension is otherwise completely silent: search.py falls
# back to the ~100-200x slower pure-Python alpha-beta with no error, no warning
# and no failing test, and the only symptom is low self-play throughput. Only
# the CUDA result gates the launch -- a missing extension warns and continues.
python -W ignore::FutureWarning - <<'PYCHECK' || {
import importlib
import sys

import torch

if not torch.cuda.is_available():
    sys.exit(1)

missing = []
for name in (
    "dama.ai.algorithmic._fast_search",
    "dama.ai.ml._fast_encode",
    "dama.ai.ml._fast_score",
):
    try:
        importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - a broken .so must not gate CUDA
        missing.append(f"{name} ({type(exc).__name__}: {exc})")

if missing:
    print("[warn] Cython accelerator(s) unavailable — falling back to pure Python:")
    for item in missing:
        print(f"[warn]   {item}")
    print("[warn] Self-play will be far slower. Rebuild with:")
    print("[warn]   cd src && python setup_cython.py build_ext --inplace")
PYCHECK
    echo ""
    echo "ERROR: CUDA is not available."
    echo ""
    echo "Troubleshooting:"
    echo "  0. Ensure the '${CONDA_ENV}' conda env exists and has torch: bash setup_conda.sh"
    echo "  1. Ensure NVIDIA GPU driver is installed on Windows (not inside WSL)"
    echo "  2. Run 'nvidia-smi' to verify GPU access"
    echo "  3. Update WSL: 'wsl --update'"
    echo "  4. Reinstall PyTorch via 'bash setup_conda.sh' — it pins the CUDA wheel"
    echo "     index (cu128) that this Blackwell GPU needs; older cu12x wheels do"
    echo "     not support it."
    exit 1
}

echo "CUDA verified. Starting training..."
echo "Config: ${SELECTED_CONFIG_PATH}"

# Build command arguments — only session-level overrides
ARGS=(--config "$SELECTED_CONFIG_PATH")

# Resume settings
if [ "$USE_RESUME_CONTINUATION" = true ]; then
    # The trainer resolves and verifies the target; --resume is omitted so the
    # config's pinned anchor remains the fallback.
    ARGS+=(--resume-continuation)
elif [ -n "$RESUME" ]; then
    ARGS+=(--resume "$RESUME")
elif [ "$RESUME_LATEST" = true ] || [ "$RESUME_LATEST" = auto ]; then
    ARGS+=(--resume-latest)
fi
if [ "$ENHANCED_STAGE" = true ]; then
    ARGS+=(--enhanced-stage --inference-depth "$ENHANCED_INFERENCE_DEPTH")
fi

# Time-based stopping (override config's time_limit if set)
if [ -n "$TRAIN_DURATION" ]; then
    ARGS+=(--train-duration "$TRAIN_DURATION")
fi

# The console log is the only record of what a session was actually launched
# with; the resolved resume target in particular is not recoverable afterwards.
echo "Trainer args: ${ARGS[*]}"
echo ""

exec python -W ignore::FutureWarning -m dama.ai.ml.trainer "${ARGS[@]}"
