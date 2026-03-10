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

HEADLESS=false
ENABLE_CAMERAS=false
ENABLE_VIDEO=false
TASK="So101-JointVelGoUp-v0"
NUM_ENVS=1024
VIDEO_LENGTH=1000
# MAX_ITERATIONS=

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_ROOT="${PROJECT_ROOT}/so101_rl"
ENV_CONFIG_PATH="${PROJECT_ROOT}/configs/baseline.yaml"
STAGED_ENV_CONFIG_PATH=""
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

    ARGS+=" --num_envs ${NUM_ENVS}"

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
        ARGS="$ARGS --video --video_length ${VIDEO_LENGTH}"
    fi

    ARGS+=" --checkpoint ${CHECKPOINT_PATH}"
    ARGS+=" --num_envs ${NUM_ENVS}"
    
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
    doctor          Print detected DISPLAY / XAUTHORITY guidance for remote SSH use
    help            Show this help message

Options:
    --task TASK              Set task name (default: ${TASK})
    --env-config PATH        YAML file for So101-LiftCube env parameters (default: ${ENV_CONFIG_PATH})
    --num-envs NUM           Set number of environments (default: ${NUM_ENVS})
    --max-iterations NUM     Set max training iterations
    --checkpoint PATH        Path to checkpoint file (required for export; used by play)
    --output-dir PATH        Reserved for custom output directory (currently not used)
    --video-length NUM       Length of recorded video in frames (default: ${VIDEO_LENGTH})
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
    $0 all --task ${TASK} --num-envs 8192 --max-iterations 10000
    $0 train --task ${TASK}
    $0 train --task So101-LiftCube-v0 --env-config configs/baseline.yaml
    $0 export --task ${TASK} --checkpoint logs/skrl/so101_rl/<run>/checkpoints/checkpoint_10000.pt
    $0 play --task ${TASK} --checkpoint logs/skrl/so101_rl/<run>/checkpoints/checkpoint_10000.pt --video --video-length 1200
    $0 train --task ${TASK} --display 0

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
            shift 2
            ;;
        --env-config)
            ENV_CONFIG_PATH="$2"
            shift 2
            ;;
        --num-envs)
            NUM_ENVS="$2"
            shift 2
            ;;
        --max-iterations)
            MAX_ITERATIONS="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --output-dir)
            CUSTOM_OUTPUT_DIR="$2"
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
        all|train|export|play|install|doctor|help)
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