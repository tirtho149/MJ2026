#!/bin/bash
#SBATCH --job-name=wsd_32b
#SBATCH --partition=nova
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=scripts/logs/wsd_32b_%j.out
#SBATCH --error=scripts/logs/wsd_32b_%j.err

# Higher-capacity model on the same WSD benchmark (Qwen2.5-32B on one H200).
set -euo pipefail
cd /work/mech-ai-scratch/tirtho/Theisis
module load cuda || true
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export HF_HOME=${HF_HOME:-/work/mech-ai-scratch/tirtho/.hf_cache}
export TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
nvidia-smi -L || true
python run_wsd.py --model Qwen/Qwen2.5-32B-Instruct --tag 32b --shots 0,1,3,5 \
    --max-model-len 4096 --tp 1
