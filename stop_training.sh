#!/bin/bash
# Stop all scheduled training processes
#
# Usage:
#   bash stop_training.sh          # Kill daemon + running session + remove cron
#   bash stop_training.sh --cron   # Only remove cron schedule
#   bash stop_training.sh --kill   # Only kill running session (daemon keeps running)
#   bash stop_training.sh --all    # Kill everything (same as no args)

# This script lives at the project root, so PROJECT_DIR is SCRIPT_DIR itself.
# scripts/runner.sh writes its PID files into the project root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
PID_FILE="$PROJECT_DIR/scheduled_runner.pid"
CHILD_PID_FILE="$PROJECT_DIR/scheduled_runner_child.pid"

stop_cron() {
    if ! command -v crontab &> /dev/null; then
        echo "crontab not available on this system (Windows). Skipping."
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
    # launcher scripts (PROCESS_TITLE in local_train.sh / train_server.sh),
    # which replaces the python cmdline and hides it from the module pattern.
    if pkill -SIGTERM -f "dama\.ai\.ml\.trainer|micro-trainer" 2>/dev/null; then
        killed=true
    fi

    if $killed; then
        echo "Waiting for trainer to exit..."
        sleep 3
        if pgrep -f "dama\.ai\.ml\.trainer|micro-trainer" >/dev/null 2>&1; then
            pkill -SIGKILL -f "dama\.ai\.ml\.trainer|micro-trainer" 2>/dev/null
        fi
        echo "Training session stopped. Daemon still running."
    else
        echo "No active training session found."
    fi
}

kill_session() {
    local killed=false

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

    # 3) Give processes time to exit gracefully
    if $killed; then
        echo "Waiting for processes to exit..."
        sleep 3
    fi

    # 4) Catch any remaining related processes via pattern match
    if pkill -SIGTERM -f "dama\.ai\.ml\.trainer|micro-trainer|script\.sh|train_server\.sh" 2>/dev/null; then
        killed=true
        sleep 2
    fi

    # 5) Force-kill anything still alive
    if pgrep -f "dama\.ai\.ml\.trainer|micro-trainer" >/dev/null 2>&1; then
        echo "Force-killing remaining trainer processes..."
        pkill -SIGKILL -f "dama\.ai\.ml\.trainer|micro-trainer" 2>/dev/null
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
        echo "Usage: bash stop_training.sh [--cron | --kill | --all]"
        exit 1
        ;;
esac
