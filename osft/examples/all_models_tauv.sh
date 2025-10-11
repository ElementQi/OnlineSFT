#!/bin/bash

set -e
set -x


MODEL_NAMES=(
    "llama31b8ins"
    "qwen25b7"
    "qwen25math7b"
)


MODEL_PATHS=(
    "your_path_to/llama31b8ins"
    "your_path_to/qwen25b7"
    "your_path_to/qwen25math7b"
)


VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export HYDRA_FULL_ERROR=1

ROOT_DIR=$(pwd)
TRAIN_FILE=$ROOT_DIR/data/deepscaler/train.parquet

VAL_PREFIX=$ROOT_DIR/data/benchmarks
MATH500_PATH=$VAL_PREFIX/math500.parquet
AIME_PATH=$VAL_PREFIX/aime.parquet
AIME25_PATH=$VAL_PREFIX/aime25.parquet
AMC_PATH=$VAL_PREFIX/amc.parquet
OLYMPIAD_PATH=$VAL_PREFIX/olympiadbench.parquet
MINERVA_PATH=$VAL_PREFIX/minerva.parquet
VAL_FILE_LIST="['$MATH500_PATH', '$AMC_PATH', '$AIME_PATH', '$AIME25_PATH', '$OLYMPIAD_PATH', '$MINERVA_PATH']"
# VAL_FILE_LIST="['$AMC_PATH']"


LR=1e-7
MAX_PROMPT_LENGTH=1024
MAX_GEN_LENGTH=3072
TASK="base-model-infos"
DATASET_NAME="dsr"
ROLLOUT_N=1
EXPERIMENT="CE-${DATASET_NAME}"
ENABLE_TRAIN_TEMP=False
TAU_S=0.6

mkdir -p ${ROOT_DIR}/logs
mkdir -p ${ROOT_DIR}/outputs


for index in ${!MODEL_NAMES[@]}; do
    BACKBONE=${MODEL_NAMES[$index]}
    MODEL_ID=${MODEL_NAMES[$index]}
    BACKBONE_PATH=${MODEL_PATHS[$index]}
    

    PROJECT_NAME="base-model-infos-${BACKBONE}"
    
    echo "============================================================"
    echo "Starting experiments for model: ${BACKBONE}"
    echo "Model Path: ${BACKBONE_PATH}"
    echo "Project Name: ${PROJECT_NAME}"
    echo "============================================================"

    for i in $(seq 0 12); do
        TAU_V=$(echo "$i * 0.1" | bc)
        DATE=$(date +"%m%d_%H%M")

        if [ $(echo "$TAU_V == 0" | bc) -eq 1 ]; then
            DO_SAMPLE=False
        else
            DO_SAMPLE=True
        fi

        MODEL="${TASK}-${BACKBONE}"
        OUTPUT_DIR="${ROOT_DIR}/outputs/base_model_infos/${MODEL}/Tauv${TAU_V}/${DATE}"

        mkdir -p ${OUTPUT_DIR}
        mkdir -p ${OUTPUT_DIR}/logs

        EXP="${TASK}-${MODEL_ID}-Tauv${TAU_V}-${DATE}"
        LOG_FILE="${OUTPUT_DIR}/logs/${EXP}.log"


        # export SWANLAB_API_KEY="your_swanlab_api_key"
        # export SWANLAB_LOG_DIR=${ROOT_DIR}/logs/swanlab/${EXP}
        # export SWANLAB_MODE=cloud
        # mkdir -p ${SWANLAB_LOG_DIR}
        # LOG_FILE="${SWANLAB_LOG_DIR}/log.txt"


        CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES} \
        python3 -m recipe.osft.generation_same_validate \
            data.train_files=$TRAIN_FILE \
            data.val_files="$VAL_FILE_LIST" \
            data.train_batch_size=128 \
            data.filter_overlong_prompts=True \
            data.max_prompt_length=${MAX_PROMPT_LENGTH} \
            data.max_response_length=${MAX_GEN_LENGTH} \
            actor_rollout_ref.model.path=${BACKBONE_PATH} \
            actor_rollout_ref.model.use_liger=False \
            actor_rollout_ref.model.use_shm=True \
            actor_rollout_ref.model.use_remove_padding=True \
            actor_rollout_ref.model.enable_gradient_checkpointing=True \
            actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
            actor_rollout_ref.actor.optim.lr=${LR} \
            actor_rollout_ref.actor.ppo_mini_batch_size=64 \
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
            actor_rollout_ref.actor.use_kl_loss=False \
            actor_rollout_ref.actor.fsdp_config.param_offload=False \
            actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
            actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
            actor_rollout_ref.rollout.name=vllm \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
            actor_rollout_ref.rollout.n=${ROLLOUT_N} \
            actor_rollout_ref.rollout.temperature=${TAU_S} \
            actor_rollout_ref.rollout.val_kwargs.temperature=${TAU_V} \
            actor_rollout_ref.rollout.val_kwargs.n=8 \
            actor_rollout_ref.rollout.val_kwargs.do_sample=${DO_SAMPLE} \
            actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
            trainer.enable_train_temperature=${ENABLE_TRAIN_TEMP} \
            trainer.rejection_sampling=False \
            trainer.logger=['console','swanlab'] \
            trainer.project_name=${PROJECT_NAME} \
            trainer.experiment_name=${EXP} \
            trainer.val_before_train=True \
            trainer.default_local_dir=${OUTPUT_DIR} \
            trainer.n_gpus_per_node=8 \
            trainer.default_hdfs_dir=null \
            trainer.nnodes=1 \
            trainer.save_freq=100 \
            trainer.rollout_data_dir=${OUTPUT_DIR}/rollout_data \
            trainer.validation_data_dir=${OUTPUT_DIR}/rollout_eval_data \
            trainer.test_freq=10 \
            +trainer.log_freq=1 \
            trainer.total_epochs=1 | tee ${LOG_FILE} || true

        echo "Run for model ${BACKBONE} with TAU_V=${TAU_V} finished. Waiting for 120 seconds..."
        sleep 30
    done
done

echo "All experiments for all models have been completed."