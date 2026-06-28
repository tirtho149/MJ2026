#!/bin/bash
#SBATCH --job-name=bangla_classify
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=scripts/logs/bangla_classify_%j.out
#SBATCH --error=scripts/logs/bangla_classify_%j.err

# Zero-shot classification eval on one GPU (vLLM-batched).  Defaults to 1000 raw
# emails; pass --data data/Bangla_Email_Dataset_Augmented.csv to score the
# augmented set, and --preprocess to classify the 13-step-cleaned text.

set -euo pipefail

cd /work/mech-ai-scratch/tirtho/Theisis

module load cuda || true
source /work/mech-ai-scratch/tirtho/.venv/bin/activate

export HF_HOME=${HF_HOME:-/work/mech-ai-scratch/tirtho/.hf_cache}
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "=== node: $(hostname) ==="
nvidia-smi -L || true

python run_classify.py --model Qwen/Qwen2.5-3B-Instruct --n-eval 1000
