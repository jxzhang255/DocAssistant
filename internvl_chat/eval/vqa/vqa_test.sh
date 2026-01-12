#!/bin/env zsh
#SBATCH -J infovqa_singleword
#SBATCH -o infovqa_singleword.out
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --mem 50GB
#SBATCH -t 40:00:00
#SBATCH --gres=gpu:a100-pcie-40gb:1
#SBATCH -w gpu21

source ~/.zshrc
# conda activate base
cd /home/jxzhang/paper_codes/InternVL/internvl_chat/eval/vqa

srun torchrun --master_port=1119 --nproc_per_node=1 evaluate_vqa.py