#!/bin/bash
#SBATCH --job-name=ft_smoke
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=scripts/logs/ft_smoke_%j.out
#SBATCH --error=scripts/logs/ft_smoke_%j.err
set -euo pipefail
cd /work/mech-ai-scratch/tirtho/Theisis
module load cuda || true
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export HF_HOME=${HF_HOME:-/work/mech-ai-scratch/tirtho/.hf_cache}
export TOKENIZERS_PARALLELISM=false
nvidia-smi -L || true
python finetune.py --smoke --model Qwen/Qwen2.5-1.5B-Instruct --epochs 1
