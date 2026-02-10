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
    local ENV_VARS="ISAAC_LAB_WORKSPACE_PATH=$WORKSPACE_PATH_VALUE"
    
    if [ -n "${X_SOCK}" ]; then
        print_info "Setting DISPLAY to :${X_SOCK} for GUI applications"
        ENV_VARS="$ENV_VARS DISPLAY=:${X_SOCK}"
        ENV_VARS="$ENV_VARS XAUTHORITY=${XAUTHORITY:-/home/${USER}/.Xauthority}"
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
    local ENV_VARS="ISAAC_LAB_WORKSPACE_PATH=$WORKSPACE_PATH_VALUE"
    
    if [ -n "${X_SOCK}" ]; then
        print_info "Setting DISPLAY to :${X_SOCK} for GUI applications"
        ENV_VARS="$ENV_VARS DISPLAY=:${X_SOCK}"
        ENV_VARS="$ENV_VARS XAUTHORITY=${XAUTHORITY:-/home/${USER}/.Xauthority}"
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
    local ENV_VARS="ISAAC_LAB_WORKSPACE_PATH=$WORKSPACE_PATH_VALUE"
    
    if [ -n "${X_SOCK}" ]; then
        print_info "Setting DISPLAY to :${X_SOCK} for GUI applications"
        ENV_VARS="$ENV_VARS DISPLAY=:${X_SOCK}"
        ENV_VARS="$ENV_VARS XAUTHORITY=${XAUTHORITY:-/home/${USER}/.Xauthority}"
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
    all             Build images, train model, play and record (full pipeline)
    install         Install the specified task into Isaac Sim
    train           Train the model only
    export          Export the trained model
    play            Play trained model and record video
    help            Show this help message

Options:
    --task TASK              Set task name (default: ${TASK})
    --num-envs NUM           Set number of environments (default: ${NUM_ENVS})
    --max-iterations NUM     Set max training iterations
    --checkpoint PATH        Path to checkpoint file for play command (auto-detects if not specified)
    --output-dir PATH        Custom output directory path (default: outputs/output_<timestamp>)
    --video-length NUM       Length of recorded video in frames (default: ${VIDEO_LENGTH})
    --headless               Run in headless mode (no GUI)
    --enable-cameras         Enable cameras in the simulation

Environment Variables:
    ISAAC_SIM_PATH           Path to local Isaac Sim installation (required)
    ISAAC_LAB_PATH           Path to local Isaac Lab installation (required)

Examples:
    $0 all --task ${TASK} --num-envs 8192 --max-iterations 10000
    $0 train --task ${TASK}
    $0 play --task ${TASK} --checkpoint outputs/output_20240101_120000/checkpoints/checkpoint_10000.pth
Output:
    All outputs (logs, models, videos) are saved to:
    ${PROJECT_ROOT}/outputs/output_<timestamp>/
    (or custom directory if --output-dir is specified)
    
    The play command will auto-detect the most recent training session
    or you can specify a checkpoint with --checkpoint
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK="$2"
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
        all|train|export|play|install|help)
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
            install_task
            exit 0
            ;;
        export)
            stage_assets
            install_task
            check_gpu
            export_model
            ;;
        train)
            stage_assets
            install_task
            check_gpu
            train_model
            ;;
        play)
            stage_assets
            install_task
            check_gpu
            play
            ;;
        all)
            stage_assets
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