#!/bin/bash
#SBATCH --job-name=fewshot_32b
#SBATCH --partition=nova
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=scripts/logs/fewshot_32b_%j.out
#SBATCH --error=scripts/logs/fewshot_32b_%j.err

# Qwen2.5-32B-Instruct (~64GB bf16) on a single H200 (140GB) — TP=1 fits with
# room for KV cache. First run downloads ~65GB to HF_HOME.
set -euo pipefail
cd /work/mech-ai-scratch/tirtho/Theisis
module load cuda || true
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export HF_HOME=${HF_HOME:-/work/mech-ai-scratch/tirtho/.hf_cache}
export TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
nvidia-smi -L || true
python run_fewshot_sweep.py --model Qwen/Qwen2.5-32B-Instruct --tag 32b \
    --shots 0,2,4,8,10,12,16 --eval-per-class 250 --tp 1 --max-model-len 8192 \
    --gpu-mem-util 0.92
