#!/bin/bash

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Defaults
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ARTIFACT_TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

HEADLESS=false
ENABLE_CAMERAS=false
ENABLE_VIDEO=false
TASK=""
NUM_ENVS=""
VIDEO_LENGTH=""
EXPERIMENT_PATH=""
NUM_EPISODES=""
NUM_VIDEOS=""
# MAX_ITERATIONS=

# Tracks which parameters were explicitly provided via CLI (to emit override warnings)
CLI_OVERRIDE_WARNINGS=()

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_ROOT="${PROJECT_ROOT}/so101_rl"
ENV_CONFIG_PATH=""
STAGED_ENV_CONFIG_PATH=""
ARTIFACTS_DIR="${PROJECT_ROOT}/artifacts/${ARTIFACT_TIMESTAMP}"
# CUSTOM_OUTPUT_DIR=""
# OUTPUT_DIR="${PROJECT_ROOT}/outputs/output_${TIMESTAMP}"
# LOGS_DIR="${OUTPUT_DIR}/logs"
# MODELS_DIR="${OUTPUT_DIR}/models"

# Functions
print_header() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

print_warn() {
    echo -e "${YELLOW}! $1${NC}"
}

print_cli_override() {
    echo -e "${YELLOW}[Warning] CLI override: $1${NC}"
}

check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        print_error "nvidia-smi not found. Please install NVIDIA drivers."
        exit 1
    fi
    
    if ! nvidia-smi &> /dev/null; then
        print_error "Cannot access NVIDIA GPU. Please check your drivers."
        exit 1
    fi
    
    print_success "GPU detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
}

discover_x11_from_processes() {
    local found=1
    local pid
    local env_dump
    local proc_name
    local discovered_display=""
    local discovered_xauthority=""

    # Prefer user-session processes first. This covers common X11 and Wayland desktops.
    while IFS= read -r pid; do
        if [ ! -r "/proc/${pid}/environ" ]; then
            continue
        fi

        env_dump=$(tr '\0' '\n' < "/proc/${pid}/environ")
        proc_name=$(ps -p "${pid}" -o comm= 2>/dev/null || true)

        discovered_display=$(echo "${env_dump}" | awk -F= '/^DISPLAY=/{print $2; exit}')
        discovered_xauthority=$(echo "${env_dump}" | awk -F= '/^XAUTHORITY=/{print $2; exit}')

        if [ -n "${discovered_display}" ] || [ -n "${discovered_xauthority}" ]; then
            print_info "Candidate GUI process: ${proc_name:-unknown} (pid ${pid})" >&2
            [ -n "${discovered_display}" ] && echo "DISPLAY=${discovered_display}" >&2
            [ -n "${discovered_xauthority}" ] && echo "XAUTHORITY=${discovered_xauthority}" >&2

            if [ -n "${discovered_display}" ] && [ -z "${DISPLAY:-}" ]; then
                export DISPLAY="${discovered_display}"
            fi
            if [ -n "${discovered_xauthority}" ] && [ -z "${XAUTHORITY:-}" ]; then
                export XAUTHORITY="${discovered_xauthority}"
            fi

            found=0
            break
        fi
    done < <(pgrep -u "$USER" -f 'gnome-shell|plasmashell|xfce4-session|Xorg|Xwayland' 2>/dev/null || true)

    return "${found}"
}

resolve_x11_environment() {
    # If user provided --display, prefer it.
    if [ -n "${X_SOCK:-}" ]; then
        export DISPLAY=":${X_SOCK}"
    fi

    # If DISPLAY/XAUTHORITY are not explicitly set, try discovering from desktop processes.
    if [ -z "${DISPLAY:-}" ] || [ -z "${XAUTHORITY:-}" ]; then
        discover_x11_from_processes || true
    fi

    # Last-resort fallback for XAUTHORITY if still unset.
    if [ -z "${XAUTHORITY:-}" ]; then
        if [ -f "${HOME}/.Xauthority" ]; then
            export XAUTHORITY="${HOME}/.Xauthority"
        elif [ -f "/home/${USER}/.Xauthority" ]; then
            export XAUTHORITY="/home/${USER}/.Xauthority"
        fi
    fi
}

get_gui_env_vars() {
    local workspace_path_value="$1"
    local env_config_path_value="$2"
    local env_vars="ISAAC_LAB_WORKSPACE_PATH=${workspace_path_value}"

    if [ -n "${env_config_path_value}" ]; then
        env_vars="${env_vars} SO101_ENV_CONFIG=${env_config_path_value}"
    fi

    resolve_x11_environment

    if [ -n "${DISPLAY:-}" ]; then
        env_vars="${env_vars} DISPLAY=${DISPLAY}"
    fi

    if [ -n "${XAUTHORITY:-}" ]; then
        env_vars="${env_vars} XAUTHORITY=${XAUTHORITY}"
    fi

    echo "${env_vars}"
}

stage_env_config() {
    local source_config="$ENV_CONFIG_PATH"

    if [ -z "$source_config" ]; then
        print_error "Environment config path is empty"
        exit 1
    fi

    if [[ "$source_config" != /* ]]; then
        source_config="${PROJECT_ROOT}/${source_config}"
    fi

    source_config="$(realpath "$source_config")"

    if [ ! -f "$source_config" ]; then
        print_error "Environment config file not found: $source_config"
        exit 1
    fi

    local config_dest_dir="$ISAAC_LAB_PATH/workspace/${TASK}/configs"
    local config_file_name
    config_file_name="$(basename "$source_config")"
    STAGED_ENV_CONFIG_PATH="${config_dest_dir}/${config_file_name}"

    mkdir -p "$config_dest_dir"
    cp "$source_config" "$STAGED_ENV_CONFIG_PATH"

    print_success "Environment config staged: ${STAGED_ENV_CONFIG_PATH}"
}

doctor_display() {
    print_header "X11 / Display Doctor"

    print_info "Current shell values"
    echo "USER=${USER}"
    echo "DISPLAY=${DISPLAY:-<unset>}"
    echo "XAUTHORITY=${XAUTHORITY:-<unset>}"
    echo "X_SOCK=${X_SOCK:-<unset>}"

    if [ -n "${XAUTHORITY:-}" ]; then
        if [ -f "${XAUTHORITY}" ]; then
            print_success "XAUTHORITY file exists: ${XAUTHORITY}"
        else
            print_warn "XAUTHORITY is set but file does not exist: ${XAUTHORITY}"
        fi
    fi

    print_info "Attempting discovery from active desktop processes"
    discover_x11_from_processes || print_warn "No GUI process with DISPLAY/XAUTHORITY env vars found"

    resolve_x11_environment

    print_info "Resolved values"
    echo "DISPLAY=${DISPLAY:-<unset>}"
    echo "XAUTHORITY=${XAUTHORITY:-<unset>}"

    if [ -n "${DISPLAY:-}" ]; then
        local sock="${DISPLAY#:}"
        sock="${sock%%.*}"
        print_info "Suggested command flags"
        echo "./scripts/run.sh train --display ${sock}"
    fi

    if [ -n "${DISPLAY:-}" ]; then
        print_info "Testing X11 cookie visibility via xauth"
        if command -v xauth >/dev/null 2>&1; then
            if xauth -f "${XAUTHORITY:-${HOME}/.Xauthority}" list "${DISPLAY}" >/dev/null 2>&1; then
                print_success "xauth can read cookie for ${DISPLAY}"
            else
                print_warn "xauth could not confirm cookie for ${DISPLAY}"
            fi
        else
            print_warn "xauth command not found; skipping cookie check"
        fi
    fi
}

# setup_directories() {
#     print_info "Setting up directories..."
    
#     # If custom output directory was specified, use it
#     if [ -n "$CUSTOM_OUTPUT_DIR" ]; then
#         # Convert to absolute path if it's relative
#         if [[ "$CUSTOM_OUTPUT_DIR" != /* ]]; then
#             CUSTOM_OUTPUT_DIR="$(cd "$(dirname "$CUSTOM_OUTPUT_DIR")" 2>/dev/null && pwd)/$(basename "$CUSTOM_OUTPUT_DIR")" || CUSTOM_OUTPUT_DIR="$(pwd)/$CUSTOM_OUTPUT_DIR"
#         fi
#         OUTPUT_DIR="$CUSTOM_OUTPUT_DIR"
#         LOGS_DIR="${OUTPUT_DIR}/logs"
#         MODELS_DIR="${OUTPUT_DIR}/models"
#         print_info "Using custom output directory: ${OUTPUT_DIR}"
#     fi
    
#     mkdir -p "${LOGS_DIR}"
#     mkdir -p "${MODELS_DIR}"
#     mkdir -p ${ISAAC_LAB_PATH}/workspace/${TASK}
#     print_success "Directories created: ${OUTPUT_DIR}"
# }

train_model() {
    # Run training script
    print_info "Starting training for task: $TASK"

    local ARGS=""
    ARGS+=" --task ${TASK}"
    if [ "${HEADLESS:-false}" = "true" ]; then
        ARGS="$ARGS --headless"
    fi
    if [ "${ENABLE_CAMERAS:-false}" = "true" ]; then
        ARGS="$ARGS --enable_cameras"
    fi

    if [ -n "${MAX_ITERATIONS}" ]; then
        ARGS="$ARGS --max_iterations ${MAX_ITERATIONS}"
    fi

    if [ -n "${CHECKPOINT_PATH}" ]; then
        ARGS+=" --checkpoint ${CHECKPOINT_PATH}"
    fi

    if [ -n "${NUM_ENVS}" ]; then
        ARGS+=" --num_envs ${NUM_ENVS}"
    fi
    ARGS+=" --artifacts_dir ${ARTIFACTS_DIR}"
    ARGS+=" hydra.run.dir=${ARTIFACTS_DIR}/hydra"

    # Copy env config into artifacts dir for reproducibility
    mkdir -p "${ARTIFACTS_DIR}"
    if [ -n "${STAGED_ENV_CONFIG_PATH}" ] && [ -f "${STAGED_ENV_CONFIG_PATH}" ]; then
        cp "${STAGED_ENV_CONFIG_PATH}" "${ARTIFACTS_DIR}/env_config.yaml"
        print_success "Env config copied to ${ARTIFACTS_DIR}/env_config.yaml"
    elif [ -n "${ENV_CONFIG_PATH}" ] && [ -f "${ENV_CONFIG_PATH}" ]; then
        cp "${ENV_CONFIG_PATH}" "${ARTIFACTS_DIR}/env_config.yaml"
        print_success "Env config copied to ${ARTIFACTS_DIR}/env_config.yaml"
    fi

    local TRAIN_COMMAND="$ISAAC_LAB_PATH/isaaclab.sh -p ${TASK_ROOT}/scripts/skrl/train.py ${ARGS}"
    print_info "Executing training command: ${TRAIN_COMMAND}"
    

    local WORKSPACE_PATH_VALUE="$ISAAC_LAB_PATH/workspace/${TASK}"
    local ENV_VARS
    ENV_VARS="$(get_gui_env_vars "${WORKSPACE_PATH_VALUE}" "${STAGED_ENV_CONFIG_PATH}")"

    if [ -n "${DISPLAY:-}" ]; then
        print_info "Using DISPLAY=${DISPLAY}"
    else
        print_warn "DISPLAY is not set. GUI windows may fail to open."
    fi
    if [ -n "${XAUTHORITY:-}" ]; then
        print_info "Using XAUTHORITY=${XAUTHORITY}"
    else
        print_warn "XAUTHORITY is not set. X11 auth may fail over SSH."
    fi
    
    /bin/bash -c "$ENV_VARS /bin/bash ${TRAIN_COMMAND}"
}

install_task() {
    print_info "Installing task: $TASK"
    $ISAAC_LAB_PATH/isaaclab.sh -p -m pip install \
        -e ${TASK_ROOT}/source/so101_rl
    print_success "Task installed: $TASK"
}

stage_assets() {
    local DEST_PATH="$ISAAC_LAB_PATH/workspace/${TASK}/assets"
    print_info "Staging assets for task: $TASK to ${DEST_PATH}"
    mkdir -p "${DEST_PATH}"
    cp -rf ${PROJECT_ROOT}/assets/. ${DEST_PATH}/
    print_success "Assets staged to ${DEST_PATH}"
}

play() {
    print_info "Running simulation with trained agent for task: $TASK"

    local ARGS=""
    ARGS+=" --task ${TASK}"
    if [ "${HEADLESS:-false}" = "true" ]; then
        ARGS="$ARGS --headless"
    fi

    if [ "${ENABLE_CAMERAS:-false}" = "true" ]; then
        ARGS="$ARGS --enable_cameras"
    fi

    if [ "${ENABLE_VIDEO:-false}" = "true" ]; then
        ARGS="$ARGS --video"
        if [ -n "${VIDEO_LENGTH}" ]; then
            ARGS="$ARGS --video_length ${VIDEO_LENGTH}"
        fi
    fi

    ARGS+=" --checkpoint ${CHECKPOINT_PATH}"
    if [ -n "${NUM_ENVS}" ]; then
        ARGS+=" --num_envs ${NUM_ENVS}"
    fi
    if [ -n "${VIDEO_LENGTH}" ]; then
        ARGS+=" --video_length ${VIDEO_LENGTH}"
    fi
    local CKPT_ROOT
    CKPT_ROOT=$(dirname "$(dirname "$(dirname "$(realpath "${CHECKPOINT_PATH}")")")") 
    ARGS+=" hydra.run.dir=${CKPT_ROOT}/hydra_play"

    local PLAY_COMMAND="$ISAAC_LAB_PATH/isaaclab.sh -p ${TASK_ROOT}/scripts/skrl/play.py ${ARGS}"
    print_info "Executing play command: ${PLAY_COMMAND}"
    
    local WORKSPACE_PATH_VALUE="$ISAAC_LAB_PATH/workspace/${TASK}"
    local ENV_VARS
    ENV_VARS="$(get_gui_env_vars "${WORKSPACE_PATH_VALUE}" "${STAGED_ENV_CONFIG_PATH}")"

    if [ -n "${DISPLAY:-}" ]; then
        print_info "Using DISPLAY=${DISPLAY}"
    else
        print_warn "DISPLAY is not set. GUI windows may fail to open."
    fi
    if [ -n "${XAUTHORITY:-}" ]; then
        print_info "Using XAUTHORITY=${XAUTHORITY}"
    else
        print_warn "XAUTHORITY is not set. X11 auth may fail over SSH."
    fi
    
    /bin/bash -c "$ENV_VARS /bin/bash ${PLAY_COMMAND}"
}

set_display() {
    print_info "Setting DISPLAY to :${X_SOCK} for GUI applications"
    export DISPLAY=":${X_SOCK}"
    export XAUTHORITY="${XAUTHORITY:-/home/${USER}/.Xauthority}"
}

evaluate_model() {
    print_info "Evaluating trained agent for experiment: ${EXPERIMENT_PATH}"

    if [ -z "${EXPERIMENT_PATH}" ]; then
        print_error "Experiment path is required for evaluation."
        exit 1
    fi

    # Convert to absolute path if relative
    if [[ "${EXPERIMENT_PATH}" != /* ]]; then
        EXPERIMENT_PATH="${PROJECT_ROOT}/${EXPERIMENT_PATH}"
    fi

    if [ ! -d "${EXPERIMENT_PATH}" ]; then
        print_error "Experiment path does not exist: ${EXPERIMENT_PATH}"
        exit 1
    fi

    # Check if env_config.yaml exists
    local ENV_CONFIG="${EXPERIMENT_PATH}/env_config.yaml"
    if [ -f "${ENV_CONFIG}" ]; then
        print_info "Found env_config.yaml: ${ENV_CONFIG}"
        ENV_CONFIG_PATH="${ENV_CONFIG}"
    else
        print_warn "No env_config.yaml found at ${ENV_CONFIG}"
    fi

    # Find task name from skrl directory
    local SKRL_DIR="${EXPERIMENT_PATH}/skrl"
    if [ ! -d "${SKRL_DIR}" ]; then
        print_error "No skrl directory found in experiment path"
        exit 1
    fi

    # Get task name (should be the only subdirectory in skrl/)
    local TASK_DIR
    TASK_DIR=$(find "${SKRL_DIR}" -mindepth 1 -maxdepth 1 -type d | head -n 1)
    if [ -z "${TASK_DIR}" ]; then
        print_error "No task directory found in ${SKRL_DIR}"
        exit 1
    fi

    local TASK_NAME
    TASK_NAME=$(basename "${TASK_DIR}")
    print_info "Detected task name: ${TASK_NAME}"

    # Check if checkpoint exists
    local CHECKPOINT="${TASK_DIR}/checkpoints/best_agent.pt"
    if [ ! -f "${CHECKPOINT}" ]; then
        print_error "Checkpoint not found: ${CHECKPOINT}"
        exit 1
    fi
    print_success "Found checkpoint: ${CHECKPOINT}"

    # Determine full task name (e.g., So101-LiftCube-v0)
    # Try to match task name pattern
    local FULL_TASK_NAME
    if [ -n "${TASK}" ]; then
        FULL_TASK_NAME="${TASK}"
    else
        # Try to construct task name from directory name
        # Convert so101_lift_cube to So101-LiftCube-v0
        FULL_TASK_NAME=$(echo "${TASK_NAME}" | sed -E 's/so101_(.*)_v([0-9]+)/So101-\U\1\E-v\2/; s/_/-/g; s/([a-z])([A-Z])/\1-\2/g; s/--/-/g')
        
        # If that didn't work, try simpler pattern
        if [[ ! "${FULL_TASK_NAME}" =~ ^So101- ]]; then
            # Default pattern
            FULL_TASK_NAME="So101-LiftCube-v0"
        fi
        print_info "Using task name: ${FULL_TASK_NAME}"
    fi

    local ARGS=""
    ARGS+=" --experiment-path ${EXPERIMENT_PATH}"
    ARGS+=" --task ${FULL_TASK_NAME}"

    if [ -n "${NUM_EPISODES}" ]; then
        ARGS+=" --num-episodes ${NUM_EPISODES}"
    fi

    if [ -n "${NUM_VIDEOS}" ]; then
        ARGS+=" --num-videos ${NUM_VIDEOS}"
    fi

    if [ "${HEADLESS:-false}" = "true" ]; then
        ARGS+=" --headless"
    fi

    local EVAL_COMMAND="$ISAAC_LAB_PATH/isaaclab.sh -p ${TASK_ROOT}/scripts/skrl/evaluate.py ${ARGS}"
    print_info "Executing evaluate command: ${EVAL_COMMAND}"

    local WORKSPACE_PATH_VALUE="$ISAAC_LAB_PATH/workspace/${FULL_TASK_NAME}"
    local ENV_VARS
    ENV_VARS="$(get_gui_env_vars "${WORKSPACE_PATH_VALUE}" "${ENV_CONFIG_PATH}")"

    if [ -n "${DISPLAY:-}" ]; then
        print_info "Using DISPLAY=${DISPLAY}"
    else
        print_warn "DISPLAY is not set. GUI windows may fail to open."
    fi
    if [ -n "${XAUTHORITY:-}" ]; then
        print_info "Using XAUTHORITY=${XAUTHORITY}"
    else
        print_warn "XAUTHORITY is not set. X11 auth may fail over SSH."
    fi

    /bin/bash -c "$ENV_VARS /bin/bash ${EVAL_COMMAND}"
}

export_model() {

    local ARGS=""
    if [ -n "${CHECKPOINT_PATH}" ]; then
        ARGS+=" --checkpoint ${CHECKPOINT_PATH}"
    else
        print_error "Checkpoint path is required for exporting the model."
        exit 1
    fi

    if [ -n "${TASK}" ]; then
        ARGS+=" --task ${TASK}"
    else
        print_error "Task name is required for exporting the model."
        exit 1
    fi
    
    if [ "${ENABLE_CAMERAS:-false}" = "true" ]; then
        ARGS="$ARGS --enable_cameras"
    fi
    
    print_info "Exporting trained model for task: $TASK"
    local CKPT_ROOT
    CKPT_ROOT=$(dirname "$(dirname "$(dirname "$(realpath "${CHECKPOINT_PATH}")")")") 
    ARGS+=" hydra.run.dir=${CKPT_ROOT}/hydra_export"

    local EXPORT_COMMAND="$ISAAC_LAB_PATH/isaaclab.sh -p ${TASK_ROOT}/scripts/skrl/export.py ${ARGS}"
    print_info "Executing export command: ${EXPORT_COMMAND}"
    
    local WORKSPACE_PATH_VALUE="$ISAAC_LAB_PATH/workspace/${TASK}"
    local ENV_VARS
    ENV_VARS="$(get_gui_env_vars "${WORKSPACE_PATH_VALUE}" "${STAGED_ENV_CONFIG_PATH}")"

    if [ -n "${DISPLAY:-}" ]; then
        print_info "Using DISPLAY=${DISPLAY}"
    else
        print_warn "DISPLAY is not set. GUI windows may fail to open."
    fi
    if [ -n "${XAUTHORITY:-}" ]; then
        print_info "Using XAUTHORITY=${XAUTHORITY}"
    else
        print_warn "XAUTHORITY is not set. X11 auth may fail over SSH."
    fi
    
    /bin/bash -c "$ENV_VARS /bin/bash ${EXPORT_COMMAND}"
}


# Install trained agent into Isaac Sim

# Stage necessary assets in the $ISAAC_SIM_PATH directory

# Run simulation with the trained agent

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Commands:
    all             Run full pipeline: stage assets, install task, check GPU, train, export, play
    install         Install the specified task package into Isaac Lab
    train           Stage assets, install task, check GPU, then train
    export          Stage assets, install task, check GPU, then export model from checkpoint
    play            Stage assets, install task, check GPU, then run policy playback
    evaluate        Run comprehensive evaluation on a trained agent from an experiment directory
    doctor          Print detected DISPLAY / XAUTHORITY guidance for remote SSH use
    help            Show this help message

Options:
    --task TASK              Set task name (required for most commands; auto-detected for evaluate)
    --env-config PATH        YAML file for So101-LiftCube env parameters (required for train/play/export)
    --experiment-path PATH   Path to experiment directory (required for evaluate)
    --num-episodes NUM       Override evaluation episode count (default: 100) [Warning emitted]
    --num-videos NUM         Override evaluation video episodes (default: 5) [Warning emitted]
    --num-envs NUM           Override num_envs from YAML config [Warning emitted]
    --max-iterations NUM     Override max training iterations (multiplied by rollouts) [Warning emitted]
    --checkpoint PATH        Path to checkpoint file (required for export; used by play)
    --output-dir PATH        Override the artifacts output directory (default: artifacts/<timestamp>/) [Warning emitted]
    --video-length NUM       Override video length in frames (downstream default used if unset) [Warning emitted]
    --video                  Enable video recording during play
    --headless               Run in headless mode (no GUI)
    --enable-cameras         Enable cameras in the simulation
    --display NUM            Set X display socket number (sets DISPLAY=:NUM for GUI apps)

Environment Variables:
    ISAAC_SIM_PATH           Path to local Isaac Sim installation (required)
    ISAAC_LAB_PATH           Path to local Isaac Lab installation (required)
    XAUTHORITY               Path to Xauthority file for GUI forwarding

Examples:
    $0 doctor
    $0 all --task So101-LiftCube-v0 --num-envs 8192 --max-iterations 10000
    $0 train --task So101-LiftCube-v0 --env-config configs/baseline.yaml
    $0 export --task So101-LiftCube-v0 --checkpoint logs/skrl/so101_rl/<run>/checkpoints/checkpoint_10000.pt
    $0 play --task So101-LiftCube-v0 --checkpoint logs/skrl/so101_rl/<run>/checkpoints/checkpoint_10000.pt --video --video-length 1200
    $0 evaluate --experiment-path artifacts/2026-03-12_09-52-10
    $0 train --task So101-LiftCube-v0 --display 0

Notes:
    The script does not currently auto-detect checkpoints.
    --output-dir is parsed but not used by the current pipeline.
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK="$2"
            CLI_OVERRIDE_WARNINGS+=("--task=${TASK} (overrides default task)")
            shift 2
            ;;
        --env-config)
            ENV_CONFIG_PATH="$2"
            shift 2
            ;;
        --num-envs)
            NUM_ENVS="$2"
            CLI_OVERRIDE_WARNINGS+=("--num-envs=${NUM_ENVS} (overrides YAML config value)")
            shift 2
            ;;
        --max-iterations)
            MAX_ITERATIONS="$2"
            CLI_OVERRIDE_WARNINGS+=("--max-iterations=${MAX_ITERATIONS} (overrides trainer.timesteps in agent config)")
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --output-dir)
            CUSTOM_OUTPUT_DIR="$2"
            CLI_OVERRIDE_WARNINGS+=("--output-dir=${CUSTOM_OUTPUT_DIR} (overrides default timestamped artifacts dir)")
            shift 2
            ;;
        --headless)
            HEADLESS="true"
            shift
            ;;
        --enable-cameras)
            ENABLE_CAMERAS="true"
            shift
            ;;
        --video-length)
            VIDEO_LENGTH="$2"
            CLI_OVERRIDE_WARNINGS+=("--video-length=${VIDEO_LENGTH} (overrides downstream default)")
            shift 2
            ;;
        --video)
            ENABLE_VIDEO="true"
            shift
            ;;
        --display)
            X_SOCK="$2"
            shift 2
            ;;
        --experiment-path)
            EXPERIMENT_PATH="$2"
            shift 2
            ;;
        --num-episodes)
            NUM_EPISODES="$2"
            CLI_OVERRIDE_WARNINGS+=("--num-episodes=${NUM_EPISODES} (overrides evaluation default)")
            shift 2
            ;;
        --num-videos)
            NUM_VIDEOS="$2"
            CLI_OVERRIDE_WARNINGS+=("--num-videos=${NUM_VIDEOS} (overrides evaluation default)")
            shift 2
            ;;
        all|train|export|play|evaluate|install|doctor|help)
            COMMAND="$1"
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_header "Training Pipeline"

    # Apply output dir override
    if [ -n "${CUSTOM_OUTPUT_DIR}" ]; then
        if [[ "${CUSTOM_OUTPUT_DIR}" != /* ]]; then
            CUSTOM_OUTPUT_DIR="${PROJECT_ROOT}/${CUSTOM_OUTPUT_DIR}"
        fi
        ARTIFACTS_DIR="${CUSTOM_OUTPUT_DIR}"
    fi

    # Emit warnings for any CLI overrides
    if [ ${#CLI_OVERRIDE_WARNINGS[@]} -gt 0 ]; then
        for warn in "${CLI_OVERRIDE_WARNINGS[@]}"; do
            print_cli_override "${warn}"
        done
    fi

    # Validate required arguments
    if [[ "${COMMAND}" != "help" && "${COMMAND}" != "doctor" && "${COMMAND}" != "evaluate" ]]; then
        if [[ -z "${TASK}" ]]; then
            print_error "--task is required for command: ${COMMAND:-<none>}"
            show_usage
            exit 1
        fi
        if [[ -z "${ENV_CONFIG_PATH}" ]]; then
            print_error "--env-config is required for command: ${COMMAND:-<none>}"
            show_usage
            exit 1
        fi
    fi

    # Validate evaluate-specific arguments
    if [[ "${COMMAND}" == "evaluate" ]]; then
        if [[ -z "${EXPERIMENT_PATH}" ]]; then
            print_error "--experiment-path is required for evaluate command"
            show_usage
            exit 1
        fi
        if [[ -n "${NUM_EPISODES}" && ! "${NUM_EPISODES}" =~ ^[0-9]+$ ]]; then
            print_error "--num-episodes must be a positive integer"
            exit 1
        fi
        if [[ -n "${NUM_EPISODES}" && "${NUM_EPISODES}" -lt 1 ]]; then
            print_error "--num-episodes must be at least 1"
            exit 1
        fi
        if [[ -n "${NUM_VIDEOS}" && ! "${NUM_VIDEOS}" =~ ^[0-9]+$ ]]; then
            print_error "--num-videos must be a positive integer"
            exit 1
        fi
        if [[ -n "${NUM_VIDEOS}" && "${NUM_VIDEOS}" -lt 0 ]]; then
            print_error "--num-videos cannot be negative"
            exit 1
        fi
        if [[ -n "${NUM_EPISODES}" && -n "${NUM_VIDEOS}" && "${NUM_VIDEOS}" -gt "${NUM_EPISODES}" ]]; then
            print_error "--num-videos cannot exceed --num-episodes"
            exit 1
        fi
    fi

    case "${COMMAND}" in
        help)
            show_usage
            exit 0
            ;;
        install)
            stage_env_config
            install_task
            exit 0
            ;;
        doctor)
            doctor_display
            exit 0
            ;;
        export)
            stage_assets
            stage_env_config
            install_task
            check_gpu
            export_model
            ;;
        train)
            stage_assets
            stage_env_config
            install_task
            check_gpu
            train_model
            ;;
        play)
            stage_assets
            stage_env_config
            install_task
            check_gpu
            play
            ;;
        evaluate)
            stage_assets
            install_task
            check_gpu
            evaluate_model
            ;;
        all)
            stage_assets
            stage_env_config
            install_task
            check_gpu
            train_model
            export_model
            play
            ;;
        *)
            print_error "No command specified"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main