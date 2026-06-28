# Reproducibility

Every reported number can be regenerated from this repo. The pipeline is
seed-controlled and the environment is pinned.

## Environment

- Python **3.11.11**, CUDA **12.8**, single NVIDIA GPU (A100 80GB / H200).
- Exact package versions are pinned in [`code/requirements.txt`](code/requirements.txt).
  The GPU stack is version-locked (vLLM ↔ torch ABI):
  `torch==2.10.0+cu128`, `vllm==0.19.1`, `transformers==5.8.1`, `peft==0.19.1`.
- On the Nova cluster this is the shared venv `/work/mech-ai-scratch/tirtho/.venv`.
  `peft` was added with `pip install --no-deps peft` so it cannot move the pinned
  torch/transformers.

## Determinism

- A single global seed (`SEED = 42`) is used everywhere. Each entry point calls
  `config.seed_everything(42)` (seeds Python / NumPy / torch + `PYTHONHASHSEED`).
- All sampling/splitting passes `random_state=42` (pandas, scikit-learn
  `train_test_split`) or a `random.Random(42)`.
- vLLM engines load with `seed=42`. Classification/WSD use greedy decoding
  (`temperature=0`) → deterministic. The uncertainty MC sampling sets
  `SamplingParams(seed=42)` so the 10-sample draw is reproducible.
- Fine-tuning sets `transformers.set_seed(42)` and `TrainingArguments(seed=42,
  data_seed=42)`.

### Known non-determinism (documented honestly)
- **Synthetic data generation** uses `temperature=0.95, seed=None` *per request*
  by design, to maximise diversity — so re-running the generator produces a
  *different* (but equally valid, equally balanced) synthetic set. The published
  dataset in `data/` is the frozen artifact used for every downstream result.
- GPU kernels (FlashAttention, cuBLAS) are not bit-exact across hardware, so
  fine-tune metrics may vary by a few tenths of a percent on a different GPU.

## Reproduce each result

```bash
cd code                      # from the repo root

# 0) instant CPU sanity (no GPU)
python smoke_test.py

# 1) (optional) regenerate the balanced dataset, then fix it to 2000/class
sbatch scripts/run_all.sbatch          # -> data/Bangla_Email_Dataset_Augmented.csv (3000/class)
python finalize_balanced.py --target 2000

# 2) few-shot sweeps (zero/few-shot in-context)
sbatch scripts/fewshot_3b.sbatch
sbatch scripts/fewshot_32b.sbatch

# 3) fine-tuning (LoRA) — the >0.70 result + t-SNE
sbatch scripts/finetune.sbatch

# 4) uncertainty quantification (MC multi-sampling)
sbatch scripts/uncertainty_3b.sbatch

# 5) Bangla word-sense disambiguation
sbatch scripts/wsd_3b.sbatch
sbatch scripts/wsd_32b.sbatch

# 6) rebuild the report tables/figures aggregation
python build_experiments_md.py
```

The shared dataset already lives in `data/`, so steps 2–5 can run directly
without re-generating it.
