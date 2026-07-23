#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

YAML_DIR="${SCRIPT_DIR}/yaml"
export EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-${YAML_DIR}/experiment.yaml}"
export RAAS_CONFIG="${RAAS_CONFIG:-${YAML_DIR}/raas.yaml}"
source "${REPO_ROOT}/examples/_common/utils.sh"
astraflow_load_experiment_env

export SERVICE_CUDA_VISIBLE_DEVICES="${SERVICE_CUDA_VISIBLE_DEVICES:-0}"
export TRAINER_MODEL0_GPUS="${TRAINER_MODEL0_GPUS:-1}"
export RAAS_HOST="${RAAS_HOST:-0.0.0.0}"
export RAAS_PORT="${RAAS_PORT:-19190}"
export ASTRAFLOW_HOST="${ASTRAFLOW_HOST:-0.0.0.0}"
export ASTRAFLOW_PORT="${ASTRAFLOW_PORT:-8000}"
export ASTRAFLOW_URL="http://127.0.0.1:${ASTRAFLOW_PORT}"
export ASTRAFLOW_RAAS_URL="http://127.0.0.1:${RAAS_PORT}"
export WEIGHT_TRANSFER_HTTP_PORT_MODEL0="${WEIGHT_TRANSFER_HTTP_PORT_MODEL0:-19861}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

TRAINER0_NPROC="$(echo "${TRAINER_MODEL0_GPUS}" | awk -F',' '{print NF}')"
astraflow_setup_env

echo "=== AstraFlow OPD smoke ==="
echo "Rollout student : Qwen/Qwen3-1.7B on GPU ${SERVICE_CUDA_VISIBLE_DEVICES}"
echo "Frozen teacher  : Qwen/Qwen3-4B"
echo "Trainer         : GPU ${TRAINER_MODEL0_GPUS}"
echo "Steps           : 2"
echo "Logs            : ${LOG_DIR}"
echo "============================"

for port in \
  "${ASTRAFLOW_PORT}" \
  "${RAAS_PORT}" \
  "${WEIGHT_TRANSFER_HTTP_PORT_MODEL0}" \
  "${MASTER_PORT_MODEL0:-29541}"; do
  if lsof -i :"${port}" >/dev/null 2>&1; then
    echo "Port ${port} is already in use; refusing to disturb another run." >&2
    exit 1
  fi
done

ASTRAFLOW_PID=""
RAAS_PID=""
TRAINER_PID=""

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  for pid in "${TRAINER_PID}" "${RAAS_PID}" "${ASTRAFLOW_PID}"; do
    if [[ -n "${pid}" ]]; then
      kill -TERM -- "-${pid}" 2>/dev/null || true
    fi
  done
  sleep 2
  for pid in "${TRAINER_PID}" "${RAAS_PID}" "${ASTRAFLOW_PID}"; do
    if [[ -n "${pid}" ]]; then
      kill -KILL -- "-${pid}" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
  exit "${status}"
}
trap cleanup EXIT INT TERM

setsid env CUDA_VISIBLE_DEVICES="" \
  python3 -u -m astraflow \
    --config "${EXPERIMENT_CONFIG}" \
    --port "${ASTRAFLOW_PORT}" \
    --host "${ASTRAFLOW_HOST}" \
    > >(tee "${LOG_DIR}/astraflow.log") 2>&1 &
ASTRAFLOW_PID=$!
sleep 5

setsid env CUDA_VISIBLE_DEVICES="${SERVICE_CUDA_VISIBLE_DEVICES}" \
  python3 -u -m astraflow.raas.server \
    --host "${RAAS_HOST}" \
    --port "${RAAS_PORT}" \
    --config "${EXPERIMENT_CONFIG}" \
    --config "${RAAS_CONFIG}" \
    --engine-id "${ENGINE_ID:-default}" \
    --astraflow-url "${ASTRAFLOW_URL}" \
    > >(tee "${LOG_DIR}/raas.log") 2>&1 &
RAAS_PID=$!
sleep 15

setsid env \
  CUDA_VISIBLE_DEVICES="${TRAINER_MODEL0_GPUS}" \
  WEIGHT_TRANSFER_HTTP_PORT="${WEIGHT_TRANSFER_HTTP_PORT_MODEL0}" \
  torchrun --nnodes 1 --nproc-per-node "${TRAINER0_NPROC}" \
    --master-addr "${MASTER_ADDR:-127.0.0.1}" \
    --master-port "${MASTER_PORT_MODEL0:-29541}" \
    examples/launch_opd_trainer.py \
    --config "${EXPERIMENT_CONFIG}" \
    --trainer trainer_model0 \
    "$@" \
    > >(tee "${LOG_DIR}/trainer_model0.log") 2>&1 &
TRAINER_PID=$!

wait "${TRAINER_PID}"
