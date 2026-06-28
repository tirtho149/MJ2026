#!/bin/bash
#SBATCH --job-name=ft_qwen
#SBATCH --partition=nova
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=scripts/logs/ft_qwen_%j.out
#SBATCH --error=scripts/logs/ft_qwen_%j.err

# Full LoRA fine-tune of Qwen2.5-3B as a 6-way Bangla email classifier on the
# balanced 12k set (stratified 80/10/10).  Reports test accuracy / macro-F1 /
# per-class + confusion + t-SNE (base vs fine-tuned).  Target: >0.70.
set -euo pipefail
cd /work/mech-ai-scratch/tirtho/Theisis
module load cuda || true
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export HF_HOME=${HF_HOME:-/work/mech-ai-scratch/tirtho/.hf_cache}
export TOKENIZERS_PARALLELISM=false
nvidia-smi -L || true
python finetune.py --model Qwen/Qwen2.5-3B-Instruct --epochs 3 --bs 16 --lr 2e-4
