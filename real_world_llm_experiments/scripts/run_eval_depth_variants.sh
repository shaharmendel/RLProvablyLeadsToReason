#!/usr/bin/env bash
# Evaluate the two additional Qwen ancestor-depth checkpoints on 15-hard.
#
# Usage:
#   ./scripts/run_eval_depth_variants.sh
#   N_SAMPLES=1000 BATCH_SIZE=16 ./scripts/run_eval_depth_variants.sh
#   SAVE_SAMPLES=1 ./scripts/run_eval_depth_variants.sh
#
# Environment overrides:
#   RUNS_DIR          where checkpoints live       (default runs)
#   N_SAMPLES         examples per model           (default 1000)
#   BATCH_SIZE        generation batch size        (default 32)
#   MAX_NEW_TOKENS    completion cap               (default 2048)
#   MAX_PROMPT_LENGTH prompt-length filter         (default 4096)
#   SEED              dataset seed                 (default 111)
#   SAVE_SAMPLES      1 => dump per-sample outputs (default 0)
#   BASE_OUTPUT_DIR   results root
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ ! -f "evaluate_simple.py" ]]; then
    echo "[error] evaluate_simple.py not found in ${REPO_ROOT}" >&2
    exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${REPO_ROOT}/logs/qwen_depth_variants_${TS}"
mkdir -p "${LOG_DIR}"

RUNS_DIR="${RUNS_DIR:-runs}"
N_SAMPLES="${N_SAMPLES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
SEED="${SEED:-111}"
SAVE_SAMPLES="${SAVE_SAMPLES:-0}"

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-evaluation_results/qwen_depth_variants_${TS}}"
mkdir -p "${BASE_OUTPUT_DIR}"

MODEL_KEYS=(
    "13_15_uniform"
    "5_10_15_uniform"
)

run_one_eval() {
    local model_key="$1"
    local model_path="$2"
    local out_dir="${BASE_OUTPUT_DIR}/${model_key}__on__15_hard"
    local log_file="${LOG_DIR}/${model_key}__on__15_hard.log"
    mkdir -p "${out_dir}"

    echo "============================================================"
    echo "[eval] $(date '+%F %T') model=qwen/${model_key} dist=15_hard"
    echo "[eval] model_path=${model_path}"
    echo "[eval] n_nodes=15 min_depth=15 max_depth=15"
    echo "[eval] output_dir=${out_dir}"
    echo "[eval] log=${log_file}"
    echo "============================================================"

    local -a cmd=(
        python evaluate_simple.py
        --model_path "${model_path}"
        --n_samples "${N_SAMPLES}"
        --batch_size "${BATCH_SIZE}"
        --max_new_tokens "${MAX_NEW_TOKENS}"
        --max_prompt_length "${MAX_PROMPT_LENGTH}"
        --output_dir "${out_dir}"
        --seed "${SEED}"
        --graph_type dag
        --min_nodes 15 --max_nodes 15
        --max_in_degree 2 --edge_probability 0.2
        --max_degree 1 --max_coefficient 1 --max_terms 2
        --probabilistic_pairwise_terms
        --single_variable_term_probability 1.0
        --pairwise_product_term_probability 0.0
        --constant_term_probability 0.6666
        --min_constant 0 --max_constant 1
        --target_sink_only
        --max_value 5000 --modulus 2
        --min_ancestor_depth 15 --max_ancestor_depth 15
        --min_start_value 0 --max_start_value 1
        --max_label_value 15 --no_fixed_label_set
        --sampling_strategy uniform_size
    )

    if [[ "${SAVE_SAMPLES}" == "1" ]]; then
        cmd+=(--save_samples)
    fi

    "${cmd[@]}" 2>&1 | tee "${log_file}"
    local status="${PIPESTATUS[0]}"
    echo "[eval] finished model=${model_key} dist=15_hard exit=${status}"
    return "${status}"
}

for model_key in "${MODEL_KEYS[@]}"; do
    model_path="${RUNS_DIR}/qwen-${model_key//_/-}"
    if [[ ! -d "${model_path}" ]]; then
        echo "[warn] skipping ${model_key}: missing directory ${model_path}" >&2
        continue
    fi
    run_one_eval "${model_key}" "${model_path}"
done

echo
echo "[done] Qwen depth-variant evaluation complete"
echo "[done] logs: ${LOG_DIR}"
echo "[done] outputs: ${BASE_OUTPUT_DIR}"
