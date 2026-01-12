# #!/bin/env zsh
# #SBATCH -J vision
# #SBATCH -o vision.out
# #SBATCH -p compute
# #SBATCH -N 1
# #SBATCH --mem 50GB
# #SBATCH -t 50:00:00
# #SBATCH --gres=gpu:a100-pcie-40gb:1
# #SBATCH -w gpu15
# source ~/.zshrc
# # conda activate base
# cd /home/jxzhang/paper_codes/InternVL/internvl_chat/internvl/train

# # # set -x

# # # PARTITION=${PARTITION:-"INTERN2"}
# # # GPUS=${GPUS:-1}
# # # GPUS_PER_NODE=${GPUS_PER_NODE:-1}
# # # # QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}
# # # NODES=$((GPUS / GPUS_PER_NODE))
# # # CPUS_PER_TASK=${CPUS_PER_TASK:-1}
# # # SRUN_ARGS=${SRUN_ARGS:-""}
# # # BATCH_SIZE=${BATCH_SIZE:-8}
# # # PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-8}
# # # GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))


# # # export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# # # export MASTER_PORT=34229
# # # export TF_CPP_MIN_LOG_LEVEL=3

# # # OUTPUT_DIR='work_dirs/internvl_chat_v1_5_phi3_3_8b_dynamic_res_finetune'

# # # if [ ! -d "$OUTPUT_DIR" ]; then
# # #   mkdir -p "$OUTPUT_DIR"
# # # fi

# # # number of gpus: 128
# # # batch size per gpu: 4
# # # gradient accumulation steps: 2
# # # total batch size: 1024
# # # epoch: 1
# srun 
torchrun --master_port=1118 internvl_chat_finetune.py \
  --model_name_or_path /home/jxzhang/paper_codes/LLM_params/InternVL2-2B \
  --conv_style "internlm2-chat" \
  --output_dir /home/jxzhang/paper_codes/InternVL/internvl_chat/shell/phi3_3_8b_dynamic/output/InternVL2B/docvqa \
  --meta_path /home/jxzhang/datasets/DUE_Benchmark/DUE_jsons/docvqa_llava.json \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 12 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --pad2square False \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone False \
  --use_backbone_lora 128 \
  --use_llm_lora 128 \
  --vision_select_layer -1 \
  --use_data_resampling False \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 10 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 1000 \
  --save_total_limit 3 \
  --learning_rate 4e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 8192 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "/home/jxzhang/paper_codes/InternVL/internvl_chat/zero_stage1_config.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "/home/jxzhang/paper_codes/InternVL/internvl_chat/shell/phi3_3_8b_dynamic/output/training_log.txt"
