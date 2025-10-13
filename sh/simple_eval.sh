set -ex

PROMPT_TYPE=$1
PROMPT_NAME=$2
MODEL_PATH=$3
MODEL_NAME=$4
MAX_TOKENS_PER_CALL=$5
OUTPUT_DIR=${MODEL_NAME}/${PROMPT_NAME}

SPLIT="test"
NUM_TEST_SAMPLE=-1
export TOKENIZERS_PARALLELISM=false

# English open datasets
DATA_NAME="gsm8k,math500"
N_SAMPLING=3

python3 -u math_eval.py \
    --model_name_or_path ${MODEL_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0.6 \
    --n_sampling ${N_SAMPLING} \
    --max_tokens_per_call ${MAX_TOKENS_PER_CALL} \
    --top_p 0.95 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs
#   --overwrite \

# Olympiad datasets
DATA_NAME="amc23,aime24"
N_SAMPLING=8

python3 -u math_eval.py \
    --model_name_or_path ${MODEL_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0.6 \
    --n_sampling ${N_SAMPLING} \
    --max_tokens_per_call ${MAX_TOKENS_PER_CALL} \
    --top_p 0.95 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs