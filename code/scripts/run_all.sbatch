#!/bin/bash
#SBATCH --job-name=bangla_all
#SBATCH --partition=nova
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=scripts/logs/bangla_all_%j.out
#SBATCH --error=scripts/logs/bangla_all_%j.err

# Full data-creation pipeline on ONE A100 (non-preemptible nova partition so an
# 18k-email generation is not killed mid-run): generate 6×3000 balanced dataset,
# zero-shot classify a balanced eval sample with the same engine, then build all
# thesis figures + tables + REPORT.md.

set -euo pipefail

cd /work/mech-ai-scratch/tirtho/Theisis

module load cuda || true
source /work/mech-ai-scratch/tirtho/.venv/bin/activate

export HF_HOME=${HF_HOME:-/work/mech-ai-scratch/tirtho/.hf_cache}
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "=== node: $(hostname) ==="
nvidia-smi -L || true

python run_all.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --per-class 3000 \
    --eval-per-class 250 \
    --max-rounds 15 \
    --overgen 1.4 \
    --max-model-len 4096 \
    --tp 1 \
    --out-dir data
