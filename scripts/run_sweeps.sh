#!/usr/bin/env bash
# run_sweeps.sh — run a fixed list of sweeps sequentially on the NAS output dir.
#
# Usage (from the project root):
#   conda run -n env_isaaclab --no-capture-output \
#     env ISAAC_LAB_PATH=/opt/isaac-sim/IsaacLab \
#     ./scripts/run_sweeps.sh [--dry-run]
#
# Each sweep runs to completion before the next one starts.  If a sweep exits
# non-zero the script stops immediately so failures are not silently swallowed.
#
# All stdout/stderr is tee'd both to the terminal and to a per-sweep log file
# under $OUTPUT_DIR.
set -euo pipefail

# ── configuration ─────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

SWEEP_SCRIPT="$SCRIPT_DIR/run.py"
CONFIGS_DIR="$PROJECT_ROOT/configs/sweeps"
OUTPUT_DIR="/mnt/nas_1/matthew-evans/so101_sim_to_real/sweeps/ablations"
LOG_DIR="$OUTPUT_DIR/logs"

SWEEPS=(
    baseline.yaml
    comparison_vision_backbone.yaml
    # ablation_grasp_phase.yaml
    # ablations_approach_phase.yaml
    # ablation_shaping.yaml
)

# ── flag parsing ──────────────────────────────────────────────────────────────

DRY_RUN_FLAG=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN_FLAG="--dry-run" ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

# ── pre-flight ────────────────────────────────────────────────────────────────

if [[ -z "${ISAAC_LAB_PATH:-}" ]]; then
    echo "[ERROR] ISAAC_LAB_PATH is not set." >&2
    echo "        export ISAAC_LAB_PATH=/path/to/IsaacLab" >&2
    exit 1
fi
if [[ ! -d "$ISAAC_LAB_PATH" ]]; then
    echo "[ERROR] ISAAC_LAB_PATH does not exist: $ISAAC_LAB_PATH" >&2
    exit 1
fi

for cfg in "${SWEEPS[@]}"; do
    if [[ ! -f "$CONFIGS_DIR/$cfg" ]]; then
        echo "[ERROR] Sweep config not found: $CONFIGS_DIR/$cfg" >&2
        exit 1
    fi
done

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# ── run sweeps ────────────────────────────────────────────────────────────────

TOTAL="${#SWEEPS[@]}"
PASS=0
FAIL=0
START_ALL="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "================================================================"
echo " run_sweeps.sh — ${TOTAL} sweep(s)  [started ${START_ALL}]"
echo " output : $OUTPUT_DIR"
echo " log dir: $LOG_DIR"
[[ -n "$DRY_RUN_FLAG" ]] && echo " mode   : DRY RUN (no training will run)"
echo "================================================================"
echo

for i in "${!SWEEPS[@]}"; do
    CFG="${SWEEPS[$i]}"
    IDX=$((i + 1))
    SWEEP_NAME="${CFG%.yaml}"
    LOG_FILE="$LOG_DIR/${SWEEP_NAME}.log"

    echo "----------------------------------------------------------------"
    echo " [$IDX/$TOTAL] $SWEEP_NAME"
    echo "   config : $CONFIGS_DIR/$CFG"
    echo "   log    : $LOG_FILE"
    echo "   started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "----------------------------------------------------------------"

    CMD=(
        "$SWEEP_SCRIPT" sweep
        --sweep "$CONFIGS_DIR/$CFG"
        --output "$OUTPUT_DIR"
    )
    [[ -n "$DRY_RUN_FLAG" ]] && CMD+=(--dry-run)

    if "$PROJECT_ROOT/scripts/run.py" sweep \
            --sweep "$CONFIGS_DIR/$CFG" \
            --output "$OUTPUT_DIR" \
            $DRY_RUN_FLAG \
            2>&1 | tee "$LOG_FILE"; then
        echo
        echo " [$IDX/$TOTAL] $SWEEP_NAME — DONE ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
        PASS=$((PASS + 1))
    else
        echo
        echo " [$IDX/$TOTAL] $SWEEP_NAME — FAILED (exit $?)" >&2
        echo "   See log: $LOG_FILE" >&2
        FAIL=$((FAIL + 1))
        exit 1
    fi
    echo
done

echo "================================================================"
echo " All sweeps complete — ${PASS}/${TOTAL} passed  [$(date -u +%Y-%m-%dT%H:%M:%SZ)]"
echo "================================================================"
