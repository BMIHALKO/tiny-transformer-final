# Tiny Transformer for English -> German Translation
**AI Course Project - Track 1: TinyReproduction**

This repository contains the final code for a **tiny reproduction** of the core ideas from **Attention Is All You Need** on an English -> German translation task.

The goal is *not* to reach state-of-the-art performance, but to **distill and verify** the behavior of a Transformer encoder-decoder on a small scale dataset, matching the spirit of Track 1 ("TinyReproduction") in the AI Course Project.

The code is designed so that:

- It **automatically downloads** the dataset via HuggingFace `datasets`.
- It **trains a SentencePiece tokenizer from scratch**.
- It trains a **small Transformer** with Noam-style learning rate warmup and label smoothing.
- It logs **loss and BLEU over time** and writes a compact **`final_metrics.json`** summary.
- A separate script **recreates the plots** used in the report from the saved logs.

---

## 1. Repository Structure

This folder contains only the files needed to reproduce the final experiments:

```text
.
├── checkpointFinal_tiny_transformer.py   # Final training & evaluation script (main entry point)
├── final_metrics.json                    # Final summary metrics from the best checkpoint
├── loss_log.json                         # Logged train/valid loss + BLEU over training
├── graphs_for_slides.py                  # Script to generate loss/BLEU graphs from logs
├── sample_translations.py                # Script or artifact with example translations
├── requirements.txt                      # Python dependencies
├── .gitignore                            # Ignore large / transient artifacts (e.g., tiny_ckpt/)
└── README.md                             # This file
```

During training, the script will create an additional directory (not tracked by git):

tiny_ckpt/
    spm.model
    spm.vocab
    sp_corpus.txt
    best.pt
    (optionally) other intermediate artifacts

This directory is **not required** to exist beforehand; it will be created automatically.

## 2. Dependencies and Environment

### **Python**
- Tested with Python 3.11
- Training runs on CPU or GPU (GPU strongly recommended but not required)

### **Install dependencies**

From this folder:
pip install -r requirements.txt

the `requirements.txt` contains:
- `torch` - model, training loop, GPU support
- `numpy` - numerical utilities
- `datasets` - HuggingFace datasets (IWSLT-style translation data)
- `evaluate` - BLEU computation via `sacrebleu`
- `sentencepiece` - unigram subword tokenizer

You may also need to follow PyTorch's official instructions for installing a GPU-enabled build if you want CUDA support.

## 3. Data and Tokenization

The script uses HuggingFace `datasets` to load English -> German translation data:
1. It first tries:
    - load_dataset("iwslt2017", "iwslt2017-en-de")
2. if that fails, it falls back to `"iwslt2017-de-en"` and flips the direction to obtain EN -> DE pairs.

If no validation/test splits are present, the script will create them via **train/validation/test splits** (10% / 10% as needed), so **no manual preprocessing is required**.

### **Tokenization**
- Uses **SentencePiece** with a unigram model:
    - shared vocabulary (source and target)
    - `vocab_size = 8000` by default
- Special token IDs follow the common convention:
    - `<unk> = 0`, `<bos> = 1`, `<eos> = 2`, `<pad> = 3`
- The tokenizer is trained on the (possibly truncated) training text only and saved under `tiny_ckpt/` as:
    - `spm.model`
    - `spm.vocab`
    - `sp_corpus.txt` (temporary training corpus)

If `spm.model` does not exist, the script will **train a new tokenizer** automatically.

## 4. Model: Tiny Transformer Architecture

The final model is a small encoder-decoder Transformer closely following the original "Attention Is All You Need" architecture, but with much smaller hyperparameters:
- `d_model = 128`
- `n_heads = 4`
- `d_ff = 512`
- `n_layers = 2` (for both encoder and decoder)
- `dropout = 0.1`
- Shared/tied output embeddings with the decoder embedding
- Sinusoidal positional encodings

Key components (all implemented in checkpointFinal_tiny_transformer.py)

- **PositionalEncoding**: sinusoidal positions added to token embeddings.
- **MultiHeadAttention**: scaled dot-prodcut attention with masking support.
- **EncoderLayer / DecoderLayer**:
    - self-attention (plus cross-attention in the decoder)
    - residual connections
    - LayerNorm
    - position-wise feed-forward networks.
- **LabelSmoothingLoss**: custom loss with smoothing (default ε = 0.1).
- **WarmupLR Scheduler**: Noam-style learning-rate schedule:
    - large LR warmup followed by inverse square-root decay.

## 5. How To Train and Evaluate

The main entry point is:

`python checkpointFinal_tiny_transformer.py`

This will:
1. Setup config and random seeds.
2. Download and prepare the EN -> DE dataset (with fallback).
3. Train or load the SentencePiece tokenizer.
4. Construct PyTorch `DataLoader`s for train/validation/test.
5. Initialize the tiny Transformer on GPU (if available) or CPU.
6. Train for a fixed number of steps using:
    - AdamW optimizer
    - Noam warmup LR schedule
    - label smoothing
    - gradient clipping
7. Every N steps (e.g., 400), evaluate on the validation set:
    - compute validation loss/token
    - derive perplexity
    - compute SacreBLEU on the validation split
    - save `best.pt` when validation loss improves
8. At the end:
    - reload the best checkpoint
    - optionally tune decoding hyperparameters (beam size, length penalty) on validation
    - compute BLEU on the test set
    - write summary metrics to `final_metrics.json`
    - write training curves to `loss_log.json`
    - generate example translations (see below)

Command-line options

The script supports several flags (names may vary slightly depending on the final version):
- `--profile`
    - Selects a model profile such as `"tiny"`, `"small"`, `"base"`.
    - For this project, `"tiny"` is the main profile of interest.
- `--steps`
    - Override the default total number of training steps.
- `--max_train`
    - Limit the number of training examples (e.g., 10000).
- `--dropout`
    - Override the dropout rate.
- `--label-smoothing`
    - Adjust the smoothing parameter for the label smoothing loss.
- `--beam-size`, `--len-pen`
    - Adjust decoding hyperparameters for beam search.

Example:

`python checkpointFinal_tiny_transformer.py --profile tiny --steps 6000 --max_train 10000`

## 6. Outputs and Files

After running `checkpointFinal_tiny_transformer.py`, you should see:

### 6.1 In the project root
- `final_metrics.json`
    - A json file containing key summary metrics for the final model (e.g., validation/test BLEU, perplexity, best step).
- `loss_log.json`
    - A json log with fields such as:
        - `train_steps`, `train_losses`
        - `valid_steps`, `valid_losses`
        - `bleu_steps`, `bleu_scores`

These are used to generate the plots in the report.

- `sample_translations.py` (or a similary named file / artifact)
    - Contains or generates example EN -> DE translations for qualitative evaluation.

### 6.2 In `tiny_ckpt/` (created automatically, not tracked in git)

- `spm.model`, `spm.vocab`, `sp_corpus.txt` - SentencePiece tokenizer artifacts
- `best.pt` - Best model weights (based on validation loss)
- Possibly additional checkpoint-related files depending on training settings

## 7. Reproducing Plots from the Report

The script `graphs_for_slides.py` reads `loss_log.json` (and optionally `final_metrics.json`) and generates the plots used in the report (e.g., train/valid loss vs. steps, BLEU vs. steps).

To regenerate these figures:

`python graphs_for_slides.py`

This will:
- Parse the training/validation curves from `loss_log.json`.
- Create line plots suitable for inclusion in the report and slides.
- Save figures as `.png` files (see inside the script for the exact output paths / filenames).

## 8. Example Translation

The pipeline also produces qualitative examples of translations (either written to a file or printed by `sample_translations.py`), showing:
- **SRC**: source English sentence
- **PRED**: predicted German translation (using beam search)
- **REF**: reference German translation

These examples are used in the report to illustrate typical success or errors of the tiny Transformer model.

## 9. Code Origin and Attribution

Per the AI Course Project guidelines, this section explains which parts of the code were:
- written by me
- adapted from prior code (e.g., course examples, standard tutorials)
- or conceptually inspired by external sources

No functions were directly copy-pasted from external repositories without modification.

### 9.1 `checkpointFinal_tiny_transformer.py`
- **Imports, configuration dataclass, and CLI parsing**
    - Written by me, following common Python patterns (`argparse`, `dataclasses`).
- **Data loading and splitting (IWSLT-style EN↔DE with fallback)**
    - Written by me using the HuggingFace datasets API.
    - The logic to try `"iwslt2017-en-de"` first and then `"iwslt2017-de-en"` and flip directions was designed and implemented by me.

- **SentencePiece training and tokenization** (`train_sentencepiece`, encode/decode helpers)
    - Written by me based on SentencePiece documentation and examples.
    - The choices of vocabulary size, special token IDs, and corpus construction were made for this project.

- **Batch/Mask utilities** (`Batch` **dataclass**, `collate_fn`, **pad masks**, **subsequent masks**)
    - Written by me, informed by lecture code and PyTorch sequence modeling examples.
    - The specific way masks are combined (padding + causal) and how `ntokens` is tracked are my own design for this project.

- **Transformer model implementation**
    - (`PositionalEncoding`, `MultiHeadAttention`, `EncoderLayer`, `DecoderLayer`, `Transformer`)
    - Written by me to follow the architecture in Vaswani et al. (2017).
    - While the structure reflects the standard Transformer components, the concrete code (naming, shape handling, masking conventions) was implemented and debugged by me for this tiny EN→DE setting.

- **Loss and scheduler** (`LabelSmoothingLoss`, `WarmupLR`)
    - Written by me, inspired by lecture material and the original paper’s description of label smoothing and the “Noam” learning rate schedule.

- **Training loop, validation, checkpointing, logging, and JSON export**
    - Written by me.
    - This includes:
        - how often validation is run,
        - how metrics are aggregated,
        - how `best.pt` is selected,
        - and how `loss_log.json` / `final_metrics.json` are structured.

- **Beam search decoding and BLEU evaluation** (`greedy_or_beam_search`, `evaluate_bleu`)
    - Written by me, using the `evaluate` library (`sacrebleu`) for BLEU computation.
    - The beam search logic (tracking log-probabilities, applying length penalty, selecting the best hypothesis) was implemented by me.

**NOTE**: This final script is an evolution of earlier checkpoint scripts written for this course. All changes and extensions for the final submission (e.g., improved logging, tunable decoding, cleaner data handling) were implemented by me.

### 9.2 `graphs_for_slides.py`

- Written entirely by me.
- Loads `loss_log.json` (and optionally `final_metrics.json`) and uses Matplotlib to generate the
figures used in the report and slides (e.g., train vs. validation loss curves, BLEU over steps).

### 9.3 `sample_translations.py`

- Written entirely by me.
- Uses the trained model and tokenizer to display or log example translations for qualitative
analysis in the report.

### 9.4 `requirements.txt`, `.gitignore`, `README.md`

- Written by me specifically for this project and this final submission, following the course
requirements for:
    - explicit dependencies,
    - ignoring large / transient artifacts,
    - and documenting how to run the code and what each file does.

## 10. How to Reproduce the Final Results (Summary)

1. Clone or unzip this repository
2. (Optional) Create and activate a virtual environment.
3. Install dependencies:
    - `pip install -r requirements.txt`
4. Run training and evaluation:
    - `python checkpointFinal_tiny_transformer.py --profile tiny`
5. After training, inspect:
    - `final_metrics.json` for summary metrics
    - `loss_log.json` for curves
    - `sample_translations.py` / output for qualitative examples.
6. Regenerate plots:
    - `python graphs_for_slides.py`

This should fully reproduce the tiny Transformer experiments used in the project report.