#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_SCRIPT="${PROJECT_ROOT}/cptl_with_social_gr/mllm_image2text.py"
LOG_DIR="${PROJECT_ROOT}/logs"

CONDA_ENV="${CONDA_ENV:-cptlsgr310_vllm}"

TEXT_GENERATION_RESTRICTIONS="${TEXT_GENERATION_RESTRICTIONS:-blind}"
DATASETS=(ETH UCY inD INTERACTION SDD)

if ! command -v conda >/dev/null 2>&1; then
    echo "Error: Conda was not found in PATH."
    exit 1
fi

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo "Error: Python script not found: ${PYTHON_SCRIPT}"
    exit 1
fi

mkdir -p "${LOG_DIR}"

OUT_LOG="${LOG_DIR}/preprocessing_mllm_image2text_output.log"
ERR_LOG="${LOG_DIR}/preprocessing_mllm_image2text_error.log"

cd "${PROJECT_ROOT}/cptl_with_social_gr"

nohup conda run -n "${CONDA_ENV}" \
    python "${PYTHON_SCRIPT}" \
    --dataset "${DATASETS[@]}" \
    --text_generation_restrictions "${TEXT_GENERATION_RESTRICTIONS}" \
    --quantization bitsandbytes \
    --save_text_descriptions \
    > "${OUT_LOG}" \
    2> "${ERR_LOG}" &

PID=$!

echo "Process PID: ${PID}"
echo "Output log: ${OUT_LOG}"
echo "Error log:  ${ERR_LOG}"

disown "${PID}"
