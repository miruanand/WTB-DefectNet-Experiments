# WTB-DefectNet — Training Codebase (Stage 1)

Code for your WTB-DefectNet paper: a texture-structure disentangling CNN with
anisotropic attention and long-tail calibration, for classifying wind-turbine-blade
surface defects on the WTBs2025 dataset (9 classes, ~7,545 images).

This is **stage 1 of the codebase**: the model, data pipeline, loss, and
sanity checks. The training loop itself (`train.py`) comes in stage 2, after
you've confirmed stage 1 runs cleanly on your machine — see "What to do next"
below.

---

## 1. What's in this folder

```
wtb_defectnet/
├── requirements.txt
├── sanity_check.py        <- RUN THIS FIRST
└── wtb/
    ├── __init__.py
    ├── config.py           <- paths, hyperparameters, seeding, device setup
    ├── dataset.py           <- reads WTBs2025's real folder layout, 60:20:20 split
    ├── model.py              <- all 7 architecture blocks + full WTBDefectNet
    ├── losses.py              <- composite loss (class-balanced focal + logit-adj + prototype)
    └── utils.py                <- metrics, checkpoint save/load, early stopping
```

## 2. What each file does

- **`config.py`** — every hyperparameter from your plan doc (batch size 32,
  AdamW wd=0.05, cosine LR with 5-epoch warmup, base LR 3e-4, 120 epochs,
  patience 20) lives here as one `Config` dataclass, plus `set_seed()` and
  `get_device()`. Change numbers here, not scattered through other files.

- **`dataset.py`** — your dataset zip unpacks to
  `<class name>/images/*.jpg` + `<class name>/labels/*.txt` (a Roboflow
  YOLO export), not a plain `ImageFolder` tree. This file indexes that real
  layout directly, ignores the bbox `.txt` labels (you're doing
  classification, not detection), and splits it **60:20:20 (train:val:test),
  per class** — so the 285-image lightning-strikes class still gets a fair
  slice in all three splits instead of being starved by a global shuffle.

- **`model.py`** — implements all 7 blocks from `Proposed_Architecture.docx`
  at the exact placements your doc specifies:
  `DSPS(stem) → Stage1(TSDB+ASA) → Stage2(TSDB+ASA+DRFB) → Stage3(TSDB+ASA+WGFR)
  → Stage4(TSDB+ASA+MSCA) → LTCP(head)`, with spatial sizes
  56→56→28→14→7 matching your plan doc's macro-architecture table exactly.

- **`losses.py`** — `L_total = L_cb + β1·L_LA + β2·L_pc` from plan doc §4.

- **`utils.py`** — macro-P/R/F1, balanced accuracy, kappa, MCC, per-class F1;
  `EarlyStopping` (keeps best-val-F1 weights); checkpoint save/load.

- **`sanity_check.py`** — builds the model, traces shapes through every
  stage, runs one backward pass, and optionally loads a real batch from your
  dataset. No training happens here — it's purely "does everything actually
  fit together" before you spend GPU-hours finding out the hard way.

## 3. Why 60:20:20 and not 5-fold CV (for now)

Your plan doc's evaluation protocol (§5) calls for stratified 5-fold CV for
the final paper numbers — that's still the right call for the headline
results, and it's coming in `train.py`. A single 60:20:20 split is what you
asked for now and is also the right first move: it's 5x cheaper to iterate
on architecture/hyperparameter bugs with one split before committing to a
5-fold run that takes 5x as long per experiment.

**Test-set discipline** (this matters for a defensible paper): use `val` for
every decision while developing — early stopping, learning-rate tuning,
"did adding DRFB help." Only run the model on `test` once, at the very end,
for the number that goes in the paper. If you re-check test repeatedly and
adjust anything based on it, it quietly becomes a second validation set and
reviewers can rightly question the reported numbers.

## 4. Setup

```bash
# 1. Unzip your dataset somewhere on the GPU machine
unzip WTBs2025.zip -d /data/WTBs2025

# 2. Create an environment (recommended)
python3 -m venv venv
source venv/bin/activate

# 3. Install PyTorch matching your CUDA version FIRST
#    (check your version: nvidia-smi, top-right corner)
#    then pick the right command from https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install the rest
pip install -r requirements.txt
```

This is a plain Python/CUDA project — no Colab magics, no Drive mounting.
It runs the same way over SSH on a lab server, a cloud GPU instance, or a
local workstation with an NVIDIA card.

## 5. What to do next (step by step)

1. **Run the sanity check.**
   ```bash
   python sanity_check.py --data_root /data/WTBs2025
   ```
   Confirm you see:
   - Shapes: `stem->(2,64,56,56)`, `stage1->(2,64,56,56)`,
     `stage2->(2,128,28,28)`, `stage3->(2,256,14,14)`,
     `stage4->(2,512,7,7)`, `logits->(2,9)`
   - Param count roughly 8–12M (your plan doc's single-GPU target)
   - `parameters with gradients: N/N OK`
   - Your real per-class counts printed, and a real batch running through
     the model without error

2. **Paste me the full console output.** If anything doesn't match — wrong
   shape, an error, a param count way outside 8–12M — send it to me exactly
   as printed and I'll fix the file, not just tell you to "check your
   install."

3. **Once it's clean, tell me and I'll give you the next batch:**
   - `train.py` — the actual training loop: AMP mixed precision, cosine LR
     with warmup, gradient accumulation, gradient clipping, checkpointing +
     resume, the weighted sampler in action, console + log-file output.
   - `evaluate.py` — runs a saved checkpoint on the held-out test set once,
     produces the full metrics table (macro-P/R/F1, balanced acc, per-class
     F1 with lightning-strikes called out, kappa, MCC) plus a confusion
     matrix plot.
   - `gradcam.py` — Grad-CAM overlays on Stage 4, for the "the model attends
     to genuine defect regions" figure your plan doc asks for.
   - After that: the 5-fold CV runner and the ablation-variant switches
     (baseline / +DSPS / +TSDB / +ASA / +LTCP / full) for your ablation study
     doc.

## 6. If something breaks on your machine

Send me: the exact command you ran, the full traceback, and your GPU
(`nvidia-smi` output). Shape mismatches, CUDA OOM (usually fixed by lowering
`batch_size` and raising `grad_accum_steps` to compensate), and
`DeformConv2d` import errors (needs `torchvision>=0.9`, already covered by
the requirements.txt pin comment) are the most common first-run issues —
all fixable in the file, not in your setup.
