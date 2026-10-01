#!/bin/bash
# Stop all scheduled training processes (macOS / Apple Silicon)
#
# macOS counterpart of stop_training.sh: the same --cron / --kill / --all
# modes, plus cleanup of the self-play workers macOS starts with 'spawn'.
#
# Usage: bash stop_training_mac.sh [--cron | --kill | --all]

# sysctl lives in /usr/sbin, which cron's default PATH (/usr/bin:/bin) leaves
# out; without it the checks below would misread this Mac.
PATH="$PATH:/usr/sbin"
# Platform guard. The Linux/WSL version is stop_training.sh.
if [ "$(uname -s)" != "Darwin" ]; then
    echo "ERROR: stop_training_mac.sh is for macOS. Use stop_training.sh on Linux/WSL." >&2
    exit 1
fi
# A terminal running under Rosetta makes every universal binary it starts run
# as x86_64 too. Re-run natively, as the other Mac scripts do.
# _DAMA_ARM64_REEXEC stops a loop if the re-exec is still translated somehow.
if [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" = 1 ]; then
    if [ -z "${_DAMA_ARM64_REEXEC:-}" ]; then
        echo "Terminal is running under Rosetta (x86_64); re-running natively as arm64..."
        export _DAMA_ARM64_REEXEC=1
        exec arch -arm64 /bin/bash "$0" "$@"
    fi
    echo "[warn] Still running under Rosetta after re-exec; continuing as x86_64."
fi

# Get the directory where this script is located
# This script lives at the project root, so PROJECT_DIR is SCRIPT_DIR itself.
# scripts/runner.sh writes its PID files into the project root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
PID_FILE="$PROJECT_DIR/scheduled_runner.pid"
CHILD_PID_FILE="$PROJECT_DIR/scheduled_runner_child.pid"
# Seconds a stopped trainer gets before it is force-killed. On SIGTERM it
# stops at the next batch, waits for an in-flight test (up to 30s) and the
# background self-play cycle (up to 60s), then saves its final checkpoint --
# far longer than the 3 seconds stop_training.sh allows, which cost that
# checkpoint. The wait ends as soon as the trainer is gone.
STOP_TIMEOUT="${STOP_TIMEOUT:-120}"

# The trainer starts its worker pools with 'spawn' on macOS, so each worker
# runs as `python -c "from multiprocessing.spawn import spawn_main..."` and
# carries neither the module path nor the process title matched below (on
# Linux the workers are forks that inherit both, and the pattern kill reaches
# them). A ProcessPoolExecutor worker never notices its parent dying -- it holds
# both ends of its task pipe -- so the project's workers watch for that
# themselves and exit within a second (dataset._start_parent_death_watchdog).
# This cleanup is the backstop for any that did not, idle but resident and
# holding unified memory. Record the trainers' direct children before anything
# is killed (afterwards launchd adopts them and they cannot be found), then
# SIGTERM whichever are still alive.
list_spawned_workers() {
    local trainer
    for trainer in $(pgrep -f "dama\.ai\.ml\.trainer|micro-trainer|micro_trainer" 2>/dev/null); do
        pgrep -P "$trainer" 2>/dev/null
    done
    return 0
}

# Workers whose trainer was already gone before this script ran -- killed with
# kill -9 or from Activity Monitor, or crashed -- have been adopted by launchd
# (PPID 1), so neither a pattern nor a trainer PID leads to them. They still
# carry the PYTHONPATH the Mac launchers export, which names this checkout, and
# `ps -E` shows a process's environment to its owner. Requiring that match
# leaves every other program's orphaned Python workers alone.
list_orphaned_workers() {
    local pid ppid cmd environ
    ps -A -o pid=,ppid=,command= 2>/dev/null | while read -r pid ppid cmd; do
        [ "$ppid" = 1 ] || continue
        case "$cmd" in *multiprocessing*) ;; *) continue ;; esac
        # The trailing space lets the last variable match like any other.
        environ="$(ps -wwE -o command= -p "$pid" 2>/dev/null) "
        case "$environ" in
            *" PYTHONPATH=${PROJECT_DIR}/src:"*|*" PYTHONPATH=${PROJECT_DIR}/src "*)
                echo "$pid"
                ;;
        esac
    done
    return 0
}

stop_spawned_workers() {
    local pid cmd seen="" stopped=0
    for pid in $1; do
        case " $seen " in *" $pid "*) continue ;; esac
        seen="$seen $pid"
        # Re-check the command line: a PID recycled since it was recorded is
        # left alone, and so are non-Python children such as the log tee. The
        # multiprocessing resource tracker ignores SIGTERM and exits by itself
        # once the workers are gone.
        cmd="$(ps -o command= -p "$pid" 2>/dev/null)"
        case "$cmd" in
            *multiprocessing.resource_tracker*) continue ;;
            *multiprocessing*) ;;
            *) continue ;;
        esac
        kill -SIGTERM "$pid" 2>/dev/null && stopped=$((stopped + 1))
    done
    if [ "$stopped" -gt 0 ]; then
        echo "Stopped $stopped leftover self-play worker(s)."
        return 0
    fi
    return 1
}

# Wait up to STOP_TIMEOUT seconds for every trainer to exit on its own.
# Returns 1 if one is still running when the time is up.
wait_for_trainers() {
    local waited=0
    while pgrep -f "dama\.ai\.ml\.trainer|micro-trainer|micro_trainer" >/dev/null 2>&1; do
        if [ "$waited" -ge "$STOP_TIMEOUT" ]; then
            return 1
        fi
        if [ "$waited" -gt 0 ] && [ $((waited % 15)) -eq 0 ]; then
            echo "  ...still saving (${waited}s)"
        fi
        sleep 1
        waited=$((waited + 1))
    done
    return 0
}

stop_cron() {
    if ! command -v crontab &> /dev/null; then
        echo "crontab not available on this system. Skipping."
        return
    fi
    if crontab -l 2>/dev/null | grep -q "runner.sh"; then
        crontab -l | grep -v "runner.sh" | crontab -
        echo "Removed runner.sh from crontab."
    else
        echo "No cron entry found for runner.sh."
    fi
}

kill_training_only() {
    local killed=false
    local workers
    workers="$(list_spawned_workers)"

    # Kill only the active training session, leave the daemon running
    if [ -f "$CHILD_PID_FILE" ]; then
        CHILD_PID=$(cat "$CHILD_PID_FILE")
        if kill -0 "$CHILD_PID" 2>/dev/null; then
            echo "Killing trainer process group (PGID $CHILD_PID)..."
            kill -SIGTERM -- -"$CHILD_PID" 2>/dev/null
            killed=true
        fi
        rm -f "$CHILD_PID_FILE"
    fi

    # Match both the module path and the setproctitle name set by the
    # launcher scripts (PROCESS_TITLE in local_train_mac.sh / local_train.sh /
    # train_server.sh), which replaces the python cmdline and hides it from
    # the module pattern.
    if pkill -SIGTERM -f "dama\.ai\.ml\.trainer|micro-trainer|micro_trainer" 2>/dev/null; then
        killed=true
    fi

    if $killed; then
        echo "Waiting for trainer to save its final checkpoint and exit (up to ${STOP_TIMEOUT}s)..."
        if ! wait_for_trainers; then
            echo "Trainer still running after ${STOP_TIMEOUT}s; force-killing it."
            pkill -SIGKILL -f "dama\.ai\.ml\.trainer|micro-trainer|micro_trainer" 2>/dev/null
        fi
    fi
    if stop_spawned_workers "$workers $(list_orphaned_workers)"; then
        killed=true
    fi

    if $killed; then
        echo "Training session stopped. Daemon still running."
    else
        echo "No active training session found."
    fi
}

kill_session() {
    local killed=false
    local timed_out=false
    local workers
    workers="$(list_spawned_workers)"

    # 1) Kill the trainer's process group (script.sh → train_server.sh → python)
    #    This runs in its own process group (set -m in runner.sh)
    if [ -f "$CHILD_PID_FILE" ]; then
        CHILD_PID=$(cat "$CHILD_PID_FILE")
        if kill -0 "$CHILD_PID" 2>/dev/null; then
            echo "Killing trainer process group (PGID $CHILD_PID)..."
            kill -SIGTERM -- -"$CHILD_PID" 2>/dev/null
            killed=true
        fi
        rm -f "$CHILD_PID_FILE"
    fi

    # 2) Kill the scheduled_runner daemon (the persistent loop + its sleep)
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Killing scheduled_runner daemon (PID $PID)..."
            kill -SIGTERM "$PID" 2>/dev/null
            killed=true
        fi
        rm -f "$PID_FILE"
    fi

    # 3) Give processes time to exit gracefully; a trainer saves its final
    #    checkpoint first
    if $killed; then
        echo "Waiting for processes to exit..."
        sleep 3
        wait_for_trainers || timed_out=true
    fi

    # 4) Catch any remaining related processes via pattern match. The Mac
    #    launcher execs the trainer once its checks pass, so it only matches
    #    while still checking -- and would start a trainer right after this.
    #    Only as run by bash: an editor or `tail -f` on the file is left alone.
    if pkill -SIGTERM -f "dama\.ai\.ml\.trainer|micro-trainer|micro_trainer|script\.sh|train_server\.sh|bash .*local_train_mac\.sh" 2>/dev/null; then
        killed=true
        sleep 2
        # A trainer reached only now (started by local_train_mac.sh rather
        # than the runner) still gets its time to save.
        $timed_out || wait_for_trainers
    fi

    # 5) Force-kill anything still alive
    if pgrep -f "dama\.ai\.ml\.trainer|micro-trainer|micro_trainer" >/dev/null 2>&1; then
        echo "Force-killing remaining trainer processes..."
        pkill -SIGKILL -f "dama\.ai\.ml\.trainer|micro-trainer|micro_trainer" 2>/dev/null
        killed=true
    fi

    # 6) Spawned workers: the stopped trainers' (recorded above), and any left
    #    behind by a trainer that died before this script ran.
    if stop_spawned_workers "$workers $(list_orphaned_workers)"; then
        killed=true
    fi

    if $killed; then
        echo "Training processes stopped."
    else
        echo "No running training processes found."
    fi
}

case "${1:-}" in
    --cron)
        stop_cron
        ;;
    --kill)
        kill_training_only
        ;;
    ''|--all)
        stop_cron
        kill_session
        ;;
    *)
        echo "Unknown option: $1"
        echo "Usage: bash stop_training_mac.sh [--cron | --kill | --all]"
        exit 1
        ;;
esac
