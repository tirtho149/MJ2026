#!/bin/bash
#SBATCH --job-name=fewshot_3b
#SBATCH --partition=nova
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=scripts/logs/fewshot_3b_%j.out
#SBATCH --error=scripts/logs/fewshot_3b_%j.err
set -euo pipefail
cd /work/mech-ai-scratch/tirtho/Theisis
module load cuda || true
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export HF_HOME=${HF_HOME:-/work/mech-ai-scratch/tirtho/.hf_cache}
export TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
nvidia-smi -L || true
python run_fewshot_sweep.py --model Qwen/Qwen2.5-3B-Instruct --tag 3b \
    --shots 0,2,4,8,10,12,16 --eval-per-class 250 --tp 1 --max-model-len 8192
