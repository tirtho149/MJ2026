#!/bin/bash
#SBATCH --job-name=bangla_generate
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=scripts/logs/bangla_generate_%j.out
#SBATCH --error=scripts/logs/bangla_generate_%j.err

# Full single-GPU (TP=1) generation of the class-balanced dataset: every class
# up-sampled to 3,000 (real + synthetic), no real data discarded.  One A100 80GB
# holds Qwen2.5-3B with room to spare — the Colab T4 OOM cannot occur here.
# Output: data/Bangla_Email_Dataset_Augmented.{csv,xlsx}

set -euo pipefail

cd /work/mech-ai-scratch/tirtho/Theisis

module load cuda || true
source /work/mech-ai-scratch/tirtho/.venv/bin/activate

export HF_HOME=${HF_HOME:-/work/mech-ai-scratch/tirtho/.hf_cache}
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "=== node: $(hostname) ==="
nvidia-smi -L || true

python run_generate.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --per-class 3000 \
    --tp 1 \
    --out-dir data
