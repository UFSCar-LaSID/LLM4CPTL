#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_SCRIPT="${PROJECT_ROOT}/cptl_with_social_gr/mllm_text2embedding.py"
LOG_DIR="${PROJECT_ROOT}/logs"

CONDA_ENV="${CONDA_ENV:-cptlsgr310_vllm}"

TEXT_GENERATION_RESTRICTIONS="${TEXT_GENERATION_RESTRICTIONS:-blind}"

DATASETS=(
    ETH
    UCY
    inD
    INTERACTION
    SDD
)

if ! command -v conda >/dev/null 2>&1; then
    echo "Error: Conda was not found in PATH."
    exit 1
fi

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo "Error: Python script not found: ${PYTHON_SCRIPT}"
    exit 1
fi

mkdir -p "${LOG_DIR}"

OUT_LOG="${LOG_DIR}/preprocessing_mllm_text2embedding_output.log"
ERR_LOG="${LOG_DIR}/preprocessing_mllm_text2embedding_error.log"

cd "${PROJECT_ROOT}/cptl_with_social_gr"

nohup conda run --no-capture-output -n "${CONDA_ENV}" \
    python -u "${PYTHON_SCRIPT}" \
    --dataset "${DATASETS[@]}" \
    --text_generation_restrictions "${TEXT_GENERATION_RESTRICTIONS}" \
    --quantization bitsandbytes \
    > "${OUT_LOG}" \
    2> "${ERR_LOG}" &

PID=$!

echo "Process PID: ${PID}"
echo "Text generation mode: ${TEXT_GENERATION_RESTRICTIONS}"
echo "Output log: ${OUT_LOG}"
echo "Error log:  ${ERR_LOG}"

disown "${PID}"
