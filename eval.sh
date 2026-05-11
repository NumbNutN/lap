#!/bin/bash
# LAP cascade-VLA eval driver (local sim).
#
# Prereqs (one-shot, before this script):
#   1. On the pod: launch the policy server.
#        kubectl exec deployment/zhaoqc-gpu-keepalive-zhaoqc-pi05-finetune-steps-25000 -- bash -lc '
#          cd /data/zhaoqc/RoboTwin/policy/lap
#          .venv/bin/python scripts/serve_policy.py \
#            --env LAP_ROBOTWIN \
#            --policy.config lap_robotwin_finetune \
#            --policy.dir checkpoints/lap_robotwin_finetune/lap_robotwin_run0/30000 \
#            --policy.type flow \
#            --port 8000
#        '
#   2. From the local host: port-forward.
#        kubectl port-forward deployment/zhaoqc-gpu-keepalive-zhaoqc-pi05-finetune-steps-25000 \
#          8001:8000
#
# Then this script can be run as:
#   bash policy/lap/eval.sh <task_name> <task_config> <train_config_name> <model_name> <seed> <gpu_id> [start_ep_id]
#
# Example:
#   bash policy/lap/eval.sh \
#     stack_blocks_two demo_clean \
#     lap_robotwin_finetune \
#     lap_robotwin_run0_step30000 \
#     0 0
#
# Notes:
# - <model_name> is a TAG used for the rollout save dir; it doesn't need to
#   match anything on the server. Convention: "<exp_name>_step<N>".
# - <gpu_id> is for the LOCAL sim renderer (RoboTwin simulator), NOT for the
#   model. The model runs on the pod's GPUs.
# - LAP doesn't use guidance_scale or scripted_grasp; those flags from the
#   pi05 eval are intentionally absent.

set -e

VERBOSE=false
while [[ "$1" == -* ]]; do
    case "$1" in
        -v|--verbose) VERBOSE=true; shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

policy_name=lap
task_name=${1:?task_name required (e.g. stack_blocks_two)}
task_config=${2:?task_config required (e.g. demo_clean)}
train_config_name=${3:?train_config_name required (e.g. lap_robotwin_finetune)}
model_name=${4:?model_name tag required (e.g. lap_robotwin_run0_step30000)}
seed=${5:?seed required}
gpu_id=${6:?gpu_id required (local renderer)}
start_ep_id=${7:-0}

# Snapshot location: persists across reboots (unlike /tmp). Override via
# EVAL_SNAPSHOT_ROOT if you want to keep it elsewhere (e.g. on a shared disk).
SNAPSHOT_ROOT="${EVAL_SNAPSHOT_ROOT:-$HOME/.cache/robotwin/eval_snapshots}"
SNAPSHOT_DIR="$SNAPSHOT_ROOT/${task_name}_seed${seed}"

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mLocal sim renderer GPU: ${gpu_id}\033[0m"
echo -e "\033[33mServer expected at $(grep -E '^server_host|^server_port' policy/lap/deploy_policy.yml | head -2)\033[0m"

cd "$(dirname "$0")/../.."   # → repo root

VERBOSE_ARGS=""
if [ "$VERBOSE" = true ]; then
    echo -e "\033[36mVerbose mode enabled\033[0m"
    VERBOSE_ARGS="--verbose true"
fi

PYTHONWARNINGS=ignore::UserWarning \
.venv/bin/python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${model_name} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --scene_snapshot "$SNAPSHOT_DIR/scenes.json" \
    --start_ep_id ${start_ep_id} \
    $VERBOSE_ARGS
