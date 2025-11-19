import math, os, random, argparse
import json
from dataclasses import dataclass
import dataclasses
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from datasets import load_dataset
import evaluate
import sentencepiece as spm
import re

import time

# -------------------------
# Hyperparameters / Config
# -------------------------

@dataclass
class Config:
    src_lang: str = "en"
    tgt_lang: str = "de"
    max_train_samples: int = 10000
    max_valid_samples: int = 2000
    max_test_samples: int = 2000
    max_len: int = 96
    vocab_size: int = 8000
    model_dir: str = "./tiny_ckpt"
    seed: int = 42
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    n_layers: int = 2
    dropout: float = 0.1
    batch_size: int = 64
    min_steps: int = 6000
    max_steps: int = 12000
    warmup_steps: int = 2000
    label_smoothing: float = 0.1
    early_stop: bool = True
    patience: int = 5
    sweep_reg: bool = False
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_eps: float = 1e-9
    beam_size: int = 4
    length_penalty: float = 0.6

cfg = Config()

PROFILES = {
    "tiny": dict(d_model = 128, n_layers = 2, n_heads = 4, d_ff = 512, dropout = 0.1),
    "small": dict(d_model = 256, n_layers = 4, n_heads = 4, d_ff = 1024, dropout = 0.1),
    "base": dict(d_model = 512, n_layers = 6, n_heads = 8, d_ff = 2048, dropout = 0.1),
}

def apply_profile(cfg, name: str):
    if name in PROFILES:
        for k, v in PROFILES[name].items():
            setattr(cfg, k, v)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", type = str, default = "tiny", choices = list(PROFILES.keys()))
    p.add_argument("--max_steps", type = int)
    p.add_argument("--min_steps", type = int)
    p.add_argument("--max_train", type = int)
    p.add_argument("--dropout", type = float)
    p.add_argument("--label_smoothing", type = float)
    p.add_argument("--beam_size", type = int)
    p.add_argument("--len_pen", type = float)
    p.add_argument("--no_early_stop", action = "store_true")
    p.add_argument("--patience", type = int)
    p.add_argument("--min_delta", type = float, default = 0.0)
    p.add_argument("--sweep_reg", action="store_true")
    p.add_argument("--pilot_steps", type = int, default = 800)

    return p.parse_args()

# -------------------------
# Utilities / Seeding
# -------------------------

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

set_seed(cfg.seed)

# -------------------------
# Data Loading (IWSLT → OPUS fallback)
# -------------------------

def load_en_de_datasets():
    # Try IWSLT2017 en-de
    try:
        print("[INFO] Loading IWSLT2017 (en-de)")
        ds = load_dataset("iwslt2017", "iwslt2017-en-de")
        flipped = False
    except Exception as e1:
        print(f"[ERROR] Could not load IWSLT2017 (en-de): {e1}")
        # Optionally: try de-en config and flip
        try:
            print("[INFO] Loading IWSLT2017 (de-en) and flipping to en→de")
            ds = load_dataset("iwslt2017", "iwslt2017-de-en")
            flipped = True
        except Exception as e2:
            raise RuntimeError(
                f"❌ Could not load IWSLT2017 (en-de or de-en). Aborting.\n"
                f"en-de error: {e1}\n"
                f"de-en error: {e2}"
            )

    # Ensure validation/test splits exist
    if "validation" not in ds or "test" not in ds:
        print("[INFO] No validation/test found; splitting train (10% each)")
        tmp = ds["train"].train_test_split(test_size=0.10, seed=cfg.seed)
        ds = {"train": tmp["train"], "test": tmp["test"]}
        tmp2 = ds["train"].train_test_split(test_size=0.10, seed=cfg.seed)
        ds["train"] = tmp2["train"]; ds["validation"] = tmp2["test"]

    # Extract pairs into simple dict
    pairs = {"train":{"src":[],"tgt":[]},
             "validation":{"src":[],"tgt":[]},
             "test":{"src":[],"tgt":[]}}

    for split in ["train","validation","test"]:
        for ex in ds[split]:
            tr = ex["translation"]
            if flipped:
                src, tgt = tr["en"], tr["de"]  # flip de-en → en-de
            else:
                src, tgt = tr["en"], tr["de"]
            pairs[split]["src"].append(src)
            pairs[split]["tgt"].append(tgt)

        print(f"[INFO] {split}: {len(pairs[split]['src'])} examples")

    return pairs

# -------------------------
# Tokenization (SentencePiece)
# -------------------------

def train_sentencepiece(src_texts: List[str], tgt_texts: List[str], vocab_size: int, workdir: str):
    os.makedirs(workdir, exist_ok=True)
    corpus_path = os.path.join(workdir, "sp_corpus.txt")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for s in src_texts + tgt_texts:
            f.write(s.strip().replace("\n", " ") + "\n")
    
    spm.SentencePieceTrainer.Train(
        input = corpus_path,
        model_prefix = os.path.join(workdir, "spm"),
        vocab_size = vocab_size,
        character_coverage = 1.0,
        model_type = "unigram",
        input_sentence_size = min(200000, len(src_texts) + len(tgt_texts)),
        shuffle_input_sentence = True,
        unk_id = 0, bos_id = 1, eos_id = 2, pad_id = 3
    )

    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(workdir, "spm.model"))

    return sp

@dataclass
class Batch:
    src: torch.Tensor
    src_mask: torch.Tensor
    tgt_in: torch.Tensor
    tgt_out: torch.Tensor
    tgt_mask: torch.Tensor
    ntokens: int

def make_pad_mask(seq: torch.Tensor, pad_idx: int):
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

def make_subsequent_mask(sz: int, device):
    mask = torch.tril(torch.ones(sz, sz, device = device)).bool()
    return mask

class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, src_texts, tgt_texts, sp, max_len, limit = None):
        self.src = src_texts[:limit] if limit else src_texts
        self.tgt = tgt_texts[:limit] if limit else tgt_texts
        self.sp = sp
        self.max_len = max_len
    
    def __len__(self):
        return len(self.src)
    
    def encode(self, text, add_bos = False, add_eos = False):
        text = text.strip().lower()
        ids = self.sp.EncodeAsIds(text)
        ids = ids[: self.max_len - (1 if add_eos else 0) - (1 if add_bos else 0)]
        if add_bos:
            ids = [1] + ids
        if add_eos:
            ids = ids + [2]
        
        return ids
    
    def __getitem__(self, idx):
        s = self.encode(self.src[idx], add_bos = False, add_eos = True)
        t = self.encode(self.tgt[idx], add_bos = True, add_eos = True)
        tgt_in = t[:-1]
        tgt_out = t[1:]

        return torch.tensor(s), torch.tensor(tgt_in), torch.tensor(tgt_out)
    
def collate_fn(batch, pad_idx = 3, device = "cpu"):
    srcs, tgts_in, tgt_out = zip(*batch)

    # Pad sequences to same length within batch
    srcs = nn.utils.rnn.pad_sequence(srcs, batch_first = True, padding_value = pad_idx)
    tgts_in = nn.utils.rnn.pad_sequence(tgts_in, batch_first = True, padding_value = pad_idx)
    tgts_out = nn.utils.rnn.pad_sequence(tgt_out, batch_first = True, padding_value = pad_idx)

    # Target sequence length
    T = tgts_in.size(1)

    # Create masks
    src_mask = make_pad_mask(srcs, pad_idx)
    tgt_pad_mask = make_pad_mask(tgts_in, pad_idx)
    subseq = make_subsequent_mask(T, srcs.device).unsqueeze(0).unsqueeze(1)
    tgt_mask = tgt_pad_mask & subseq

    # Count non-padding tokens for loss normalization
    ntokens = (tgts_out != pad_idx).sum().item()

    return Batch(srcs, src_mask, tgts_in, tgts_out, tgt_mask, ntokens)

def decode_until(ids, stop_ids = (2,3)):
    out = []
    for t in ids:
        if t in stop_ids:
            break
        out.append(t)
    return out

def looks_like_text(s: str) -> bool:
    s = s.strip()
    if len(s) < 8:
        return False
    letters = sum(c.isalpha() for c in s)
    digits = sum(c.isdigit() for c in s)

    # at least 20 characters and at least half of the chars are letters'
    if letters < max(10, int(0.5 * len(s))):
        return False
    # not too many digits
    if digits > 0.3 * (letters + digits + 1):
        return False
    # avoid strings that are mostly punctuation
    if re.fullmatch(r'[\W_]+', s):
        return False
    
    return True

def pick_nice_examples(triples, k = 5):
    nice = []
    for src, pred, ref in triples:
        if looks_like_text(src) and looks_like_text(ref):
            nice.append((src, pred, ref))
            if len(nice) == k:
                break
    
    # if we dont find enough, fill with the first ones
    if len(nice) < k:
        nice += triples[:max(0, k - len(nice))]
    
    return nice[:k]

# -------------------------
# Model: Tiny Transformer
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0.1, max_len = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x * math.sqrt(x.size(-1))
        x = x + self.pe[:, :x.size(1)]

        return self.dropout(x)
    
def scaled_dot_attn(q, k, v, mask = None):
    d_k = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    
    attn = F.softmax(scores, dim = -1)

    return attn @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def split_heads(self, x):
        B, L, D = x.size()
        x = x.view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        return x
    
    def combine_heads(self, x):
        B, h, L, D_k = x.size()
        x = x.transpose(1, 2).contiguous().view(B, L, h * D_k)

        return x
    
    def forward(self, q, k, v, mask = None):
        q = self.split_heads(self.w_q(q))
        k = self.split_heads(self.w_k(k))
        v = self.split_heads(self.w_v(v))
        if mask is not None:
            mask = mask.expand(-1, self.n_heads, -1, -1)
        
        x = scaled_dot_attn(q, k, v, mask)
        x = self.combine_heads(x)

        return self.w_o(self.dropout(x))

class PositionwiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout = 0.1):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.lin2(self.dropout(F.relu(self.lin1(x))))
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mem, tgt_mask, src_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, mem, mem, src_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))

        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, dropout = 0.1, pad_idx = 3):
        super().__init__()
        self.pad_idx = pad_idx
        self.src_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos = PositionalEncoding(d_model, dropout)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.proj = nn.Linear(d_model, vocab_size, bias = False)
        self.proj.weight = self.tgt_emb.weight


    def encode(self, src, src_mask):
        x = self.pos(self.src_emb(src))
        for layer in self.enc_layers:
            x = layer(x, src_mask)
        
        return x
    
    def decode(self, tgt_in, mem, tgt_mask, src_mask):
        x = self.pos(self.tgt_emb(tgt_in))
        for layer in self.dec_layers:
            x = layer(x, mem, tgt_mask, src_mask)
        
        return x
    
    def forward(self, batch):
        mem = self.encode(batch.src, batch.src_mask)
        dec = self.decode(batch.tgt_in, mem, batch.tgt_mask, batch.src_mask)
        logits = self.proj(dec)

        return logits

# -------------------------
# Loss: Label Smoothing
# -------------------------
class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, padding_idx = 3, smoothing = 0.1):
        super().__init__()
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.vocab_size = vocab_size

    def forward(self, logits, target):
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (self.vocab_size - 2))
            ignore = (target == self.padding_idx)
            target = target.clone()
            target[ignore] = 0
            true_dist.scatter_(2, target.unsqueeze(2), 1.0 - self.smoothing)
            true_dist.masked_fill_(ignore.unsqueeze(2), 0.0)
        
        log_probs = F.log_softmax(logits, dim = -1)
        loss = -(true_dist * log_probs).sum(dim = 2)
        denom = (~ignore).sum().clamp_min(1)

        return loss.sum() / denom

class WarmupLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps, last_epoch = -1):
        self.d_model = d_model
        self.warmup = warmup_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        step = max(1, self.last_epoch + 1)
        scale = (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup ** -1.5))

        return [base_lr * scale for base_lr in self.base_lrs]

# -------------------------
# Training / Evaluation
# -------------------------
def train_step(model, batch, optimizer, scheduler, criterion, device, scaler):
    model.train()
    optimizer.zero_grad(set_to_none = True)
    with autocast(enabled = (device == "cuda")):
        logits = model(batch)
        loss = criterion(logits, batch.tgt_out)
    scaler.scale(loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    
    return loss.item()

def greedy_or_beam_search(model, sp, src, max_len, beam_size = 4, len_pen = 0.6, device = "cpu"):
    model.eval()
    pad, bos, eos = 3, 1, 2
    with torch.no_grad():
        src_mask = make_pad_mask(src, pad)
        mem = model.encode(src, src_mask)
        beams = [(0.0, torch.tensor([[bos]], device = device, dtype = torch.long))]
        finished = []
        
        for _ in range(max_len):
            new_beams = []
            for logp, seq in beams:
                if seq[0, -1].item() == eos:
                    finished.append((logp, seq))
                    continue
                T = seq.size(1)
                tgt_mask = make_pad_mask(seq, pad) & make_subsequent_mask(T, device).unsqueeze(0).unsqueeze(1)
                dec = model.decode(seq, mem, tgt_mask, src_mask)
                logits = model.proj(dec[:, -1:, :])
                probs = F.log_softmax(logits, dim = -1).squeeze(1)
                topk = torch.topk(probs, beam_size, dim = -1)
                for k in range(beam_size):
                    tok = topk.indices[0, k].unsqueeze(0).unsqueeze(0)
                    score = topk.values[0, k].item()
                    new_seq = torch.cat([seq, tok], dim = 1)
                    new_beams.append((logp + score, new_seq))
            new_beams.sort(key = lambda x: x[0], reverse = True)
            beams = new_beams[:beam_size]
            if len(finished) >= beam_size:
                break
        if not finished:
            finished = beams
        
        def lp(l): return ((5 + l) ** len_pen) / ((5 + 1) ** len_pen)
        
        scored = [(logp / lp(seq.size(1)), seq) for logp, seq in finished]
        scored.sort(key = lambda x: x[0], reverse = True)
        best = scored[0][1][0].tolist()
        
        if eos in best:
            best = best[1: best.index(eos)]
        else:
            best = best[1:]
        
        return sp.DecodeIds(best)

SACREBLEU = evaluate.load("sacrebleu")

def evaluate_bleu(model, sp, dl, device = "cpu"):
    preds, refs, srcs = [], [], []
    
    for batch in dl:
        for i in range(batch.src.size(0)):
            src_ids = batch.src[i].tolist()
            src_text = sp.DecodeIds(decode_until(src_ids, stop_ids = (2, 3)))
            srcs.append(src_text.strip())

            src = batch.src[i: i + 1].to(device)
            pred = greedy_or_beam_search(
                model, sp, src, cfg.max_len, beam_size = cfg.beam_size, 
                len_pen = cfg.length_penalty, device = device
            )
            tgt_ids = batch.tgt_out[i].tolist()
            
            if 2 in tgt_ids:
                tgt_ids = tgt_ids[: tgt_ids.index(2)]
            
            tgt_text = sp.DecodeIds([t for t in tgt_ids if t not in (0,3)])
            preds.append(pred.strip())
            refs.append([tgt_text.strip()])
    bleu = SACREBLEU.compute(predictions = preds, references = refs, force = True)["score"]
    return bleu, list(zip(srcs, preds, [r[0] for r in refs]))

def sweep_regularization(base_cfg, data, sp, device, collate, grids=None, pilot_steps=800):
    if grids is None:
        grids = {"dropout": [0.1, 0.2], "label_smoothing": [0.05, 0.1, 0.15]}
    
    best = (None, None, float("inf"))
    for d in grids["dropout"]:
        for ls in grids["label_smoothing"]:
            tmp_cfg = dataclasses.replace(base_cfg, dropout=d, label_smoothing=ls)

            coll = collate

            tmp_model = Transformer(
                vocab_size = tmp_cfg.vocab_size,
                d_model = tmp_cfg.d_model,
                n_heads = tmp_cfg.n_heads,
                d_ff = tmp_cfg.d_ff,
                n_layers = tmp_cfg.n_layers,
                dropout = tmp_cfg.dropout,
                pad_idx=3,
            ).to(device)

            opt = torch.optim.AdamW(tmp_model.parameters(), lr=1.0,
                                    betas = (tmp_cfg.adam_beta1, tmp_cfg.adam_beta2), eps=tmp_cfg.adam_eps)
            sched = WarmupLR(opt, d_model=tmp_cfg.d_model, warmup_steps=tmp_cfg.warmup_steps)
            crit = LabelSmoothingLoss(vocab_size=tmp_cfg.vocab_size, padding_idx=3, smoothing=ls)

            train_src, train_tgt = data["train"]["src"], data["train"]["tgt"]
            valid_src, valid_tgt = data["validation"]["src"], data["validation"]["tgt"]

            train_ds = SeqDataset(train_src, train_tgt, sp, tmp_cfg.max_len)
            valid_ds = SeqDataset(valid_src, valid_tgt, sp, tmp_cfg.max_len)

            train_dl = DataLoader(train_ds, batch_size=tmp_cfg.batch_size, shuffle=True, collate_fn=coll)
            valid_dl = DataLoader(valid_ds, batch_size=tmp_cfg.batch_size, shuffle=False, collate_fn=coll)

            tmp_model.train()
            step = 0
            for batch in train_dl:
                logits = tmp_model(batch)
                loss = crit(logits, batch.tgt_out) / max(1, batch.ntokens)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(tmp_model.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step >= pilot_steps:
                    break

            tmp_model.eval()
            vloss_total, vntok = 0.0, 0
            with torch.no_grad():
                for vb in valid_dl:
                    logits = tmp_model(vb)
                    vloss_total += crit(logits, vb.tgt_out).item() * vb.ntokens
                    vntok += vb.ntokens
                
            dev_loss = vloss_total / max(1, vntok)
            print(f"[SWEEP] dropout={d} ls={ls} → dev loss/token={dev_loss:.4f}")

            if dev_loss < best[2]:
                best = (d, ls, dev_loss)
    print(f"[Sweep] best: dropout={best[0]} ls={best[1]} dev loss/token={best[2]:.4f}")
    return best[0], best[1]

def tune_decoding(model, sp, dev_dl, device="cpu", beams=(3,4,5,6), lens=(0.2,0.4,0.6,0.8,1.0)):
    best = (cfg.beam_size, cfg.length_penalty, -1.0)
    for b in beams:
        for lp in lens:
            cfg.beam_size, cfg.length_penalty = b, lp
            bleu, _ = evaluate_bleu(model, sp, dev_dl, device)
            if bleu > best[2]:
                best = (b, lp, bleu)
    cfg.beam_size, cfg.length_penalty = best[0], best[1]
    print(f"[DECODE TUNE] best beam={best[0]} len_pen={best[1]} → dev BLEU={best[2]:.2f}")
    return best

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg.model_dir, exist_ok = True)

    args = parse_args()
    apply_profile(cfg, args.profile)
    if args.max_steps is not None: cfg.max_steps = args.max_steps
    if args.min_steps is not None: cfg.min_steps = args.min_steps
    if args.patience is not None: cfg.patience = args.patience
    cfg.early_stop = True
    if args.no_early_stop: cfg.early_stop = False
    if args.sweep_reg: cfg.sweep_reg = True
    if args.max_train is not None: cfg.max_train_samples = args.max_train
    if args.dropout is not None: cfg.dropout = args.dropout
    if args.label_smoothing is not None: cfg.label_smoothing = args.label_smoothing
    if args.beam_size is not None: cfg.beam_size = args.beam_size
    if args.len_pen is not None: cfg.length_penalty = args.len_pen
    pilot_steps = args.pilot_steps
    min_delta = args.min_delta
    print(f"[CONFIG] {cfg}")

    data = load_en_de_datasets()

    def sub(x, n):
        return x[:n]
    
    train_src = sub(data["train"]["src"], cfg.max_train_samples)
    train_tgt = sub(data["train"]["tgt"], cfg.max_train_samples)
    valid_src = sub(data["validation"]["src"], cfg.max_valid_samples)
    valid_tgt = sub(data["validation"]["tgt"], cfg.max_valid_samples)
    test_src  = sub(data["test"]["src"], cfg.max_test_samples)
    test_tgt  = sub(data["test"]["tgt"], cfg.max_test_samples)
    sp = train_sentencepiece(train_src, train_tgt, cfg.vocab_size, cfg.model_dir)
    train_ds = SeqDataset(train_src, train_tgt, sp, cfg.max_len)
    valid_ds = SeqDataset(valid_src, valid_tgt, sp, cfg.max_len)
    test_ds  = SeqDataset(test_src,  test_tgt,  sp, cfg.max_len)
    collate = lambda b: collate_fn(b, pad_idx = 3, device = device)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)
    valid_dl = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)
    test_dl  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)

    if cfg.sweep_reg:
        print(f"[INFO] Starting lightweight reg sweep (pilot training)")
        data_small = {
            "train": {"src": train_src, "tgt": train_tgt}, 
            "validation": {"src": valid_src, "tgt": valid_tgt}
        }
        best_d, best_ls = sweep_regularization(cfg, data_small, sp, device, collate,
                                               grids={"dropout": [0.1, 0.2], "label_smoothing": [0.05, 0.1, 0.15]},
                                               pilot_steps=pilot_steps)
        cfg.dropout, cfg.label_smoothing = best_d, best_ls
        print(f"[INFO] Using best regs → dropout={cfg.dropout}, label_smoothing={cfg.label_smoothing}")

    model = Transformer(
        vocab_size = cfg.vocab_size,
        d_model = cfg.d_model, n_heads = cfg.n_heads, d_ff = cfg.d_ff,
        n_layers = cfg.n_layers, dropout = cfg.dropout, pad_idx = 3
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr = 1.0,
        betas = (cfg.adam_beta1, cfg.adam_beta2),
        eps = cfg.adam_eps
    )
    sched = WarmupLR(opt, d_model = cfg.d_model, warmup_steps = cfg.warmup_steps)
    crit = LabelSmoothingLoss(vocab_size = cfg.vocab_size, padding_idx = 3, smoothing = cfg.label_smoothing)

    train_steps, train_losses = [], []
    valid_steps, valid_losses = [], []
    bleu_steps, bleu_scores = [], []

    step = 0
    tok_counter, time_ref = 0, time.time()
    best_valid = float("inf")
    epochs_no_improve = 0
    scaler = GradScaler(enabled = (device == "cuda"))
    model.train()
    while step < cfg.max_steps:
        for batch in train_dl:
            batch.src = batch.src.to(device)
            batch.src_mask = batch.src_mask.to(device)
            batch.tgt_in = batch.tgt_in.to(device)
            batch.tgt_out = batch.tgt_out.to(device)
            batch.tgt_mask = batch.tgt_mask.to(device)

            loss = train_step(model, batch, opt, sched, crit, device, scaler)
            step += 1
            tok_counter += batch.ntokens

            if step % 100 == 0:
                dt = time.time() - time_ref
                tps = tok_counter / max(1e-6, dt)
                print(f"[step {step}] train loss: {loss:.4f} | tok/s: {tps:.0f}")
                tok_counter, time_ref = 0, time.time()
                train_steps.append(step)
                train_losses.append(loss)

            if step % 400 == 0:
                model.eval()
                with torch.no_grad():
                    vloss_total, vntok, nll_total = 0.0, 0, 0.0
                    for vb in valid_dl:
                        vb.src = vb.src.to(device)
                        vb.src_mask = vb.src_mask.to(device)
                        vb.tgt_in = vb.tgt_in.to(device)
                        vb.tgt_out = vb.tgt_out.to(device)
                        vb.tgt_mask = vb.tgt_mask.to(device)

                        logits = model(vb)
                        vloss_total +=  crit(logits, vb.tgt_out).item() * vb.ntokens
                        vntok += vb.ntokens

                        V = logits.size(-1)
                        log_probs = F.log_softmax(logits, dim = -1)
                        nll_total += F.nll_loss(
                            log_probs.view(-1, V),
                            vb.tgt_out.view(-1),
                            ignore_index = 3,
                            reduction='sum'
                        ).item()

                vloss = vloss_total / max(1, vntok)
                avg_nll = nll_total / max(1, vntok)
                
                ppl = math.exp(min(20.0, avg_nll))
                
                print(f"         valid loss/token: {vloss:.6f} | PPL: {ppl:.3f}")
                valid_steps.append(step)
                valid_losses.append(vloss)

                bleu, _ = evaluate_bleu(model, sp, valid_dl, device)
                print(f"         valid BLEU: {bleu:.2f}")
                bleu_steps.append(step)
                bleu_scores.append(bleu)
                
                if vloss + min_delta < best_valid:
                    best_valid = vloss
                    epochs_no_improve = 0
                    torch.save(model.state_dict(), os.path.join(cfg.model_dir, "best.pt"))
                    print(f"[BEST] step={step} dev={best_valid:.6f} (saved best.pt)")
                else:
                    epochs_no_improve += 1
                
                if cfg.early_stop and step >= cfg.min_steps and epochs_no_improve >= (cfg.patience or 0):
                    print(f"[EARLY STOP] no improvement for {cfg.patience} evals at step {step}. Best valid={best_valid:.6f}")
                    step = cfg.max_steps
                
                model.train()
            
            if step >= cfg.max_steps:
                break
    if os.path.exists(os.path.join(cfg.model_dir, "best.pt")):
        model.load_state_dict(torch.load(os.path.join(cfg.model_dir, "best.pt"), map_location = device))
    
    
    
    tune_decoding(model, sp, valid_dl, device)
    
    bleu, samples = evaluate_bleu(model, sp, DataLoader(test_ds, batch_size = 16, collate_fn = collate), device)
    print("\n--- PRELIMINARY RESULT ---")
    print({"sacrebleu": bleu})
    print("\n--- SAMPLE TRANSLATIONS ---")
    bleu_steps.append(step)
    bleu_scores.append(bleu)

    nice = pick_nice_examples(samples, k=5)
    for i, (src, pred, ref) in enumerate(nice, 1):
        print(f"[{i}] SRC:  {src}\n     PRED: {pred}\n     REF:  {ref}\n")
    
    final_dir = os.path.join(cfg.model_dir, "FinalCheckpoint")
    os.makedirs(final_dir, exist_ok=True)
    
    with open(os.path.join(final_dir, "final_metrics.json"), "w", encoding = "utf-8") as f:
        json.dump({
            "train_steps": train_steps,
            "train_losses": train_losses,
            "valid_steps": valid_steps,
            "valid_losses": valid_losses,
            "bleu_steps": bleu_steps,
            "bleu_scores": bleu_scores,
            "final_bleu": bleu,
            "best_valid_loss": best_valid
        }, f, ensure_ascii = False, indent = 2)
    
    with open(os.path.join(final_dir, "sample_translations.txt"), "w", encoding= "utf-8") as f:
        for i, (src, pred, ref) in enumerate(nice, 1):
            f.write(f"[{i}] SRC:  {src}\n     PRED: {pred}\n     REF:  {ref}\n\n")
    
    print(f"\n✅ Final results and translations saved to: {final_dir}")

if __name__ == "__main__":
    main()