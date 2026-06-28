#!/bin/bash
#SBATCH --job-name=bangla_smoke
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=scripts/logs/bangla_smoke_%j.out
#SBATCH --error=scripts/logs/bangla_smoke_%j.err

# End-to-end single-GPU smoke test: Stage A (CPU sanity) + Stage B (tiny vLLM
# generation with the 1.5B model, then OOM-safe engine teardown).  Confirms the
# corpus, prompts, parsers, dedup and the save path all work on ONE Nova GPU
# before launching the full run.  Downloads the model to HF_HOME on first use.

set -euo pipefail

cd /work/mech-ai-scratch/tirtho/Theisis        # SLURM copies $0; dirname "$0" is unreliable

module load cuda || true
source /work/mech-ai-scratch/tirtho/.venv/bin/activate   # pinned torch 2.10.0 / vLLM 0.19.1

export HF_HOME=${HF_HOME:-/work/mech-ai-scratch/tirtho/.hf_cache}
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "=== node: $(hostname) ==="
nvidia-smi -L || true
python -c "import torch; print('cuda available:', torch.cuda.is_available())"

echo "=== smoke test (Qwen2.5-1.5B, all 6 classes, 4 synthetic/class) ==="
python smoke_test.py
