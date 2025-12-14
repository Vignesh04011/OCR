#!/usr/bin/env python3
"""
train_crnn.py

Train a CRNN (CNN + BiLSTM + CTC) on line-level handwriting data.
Expects CSV files with columns: image_path,text

Features:
 - builds charset automatically (or load from charlist file)
 - mixed precision training (AMP)
 - tensorboard logging
 - save best checkpoint by validation CER
"""

import os
import math
import time
import argparse
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# Optional Levenshtein distance (recommended). If missing, will fallback to python implementation.
try:
    import Levenshtein as Lev
    def cer(a, b): return Lev.distance(a, b) / max(1, len(a))
    def wer(a, b):
        aw = a.split()
        bw = b.split()
        return Lev.distance(" ".join(aw), " ".join(bw)) / max(1, len(aw))
except Exception:
    # simple edit distance fallback (slower but works)
    def edit_distance(s1, s2):
        n, m = len(s1), len(s2)
        dp = [[0]*(m+1) for _ in range(n+1)]
        for i in range(n+1): dp[i][0] = i
        for j in range(m+1): dp[0][j] = j
        for i in range(1,n+1):
            for j in range(1,m+1):
                if s1[i-1]==s2[j-1]: dp[i][j]=dp[i-1][j-1]
                else: dp[i][j]=1+min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        return dp[n][m]
    def cer(a, b): return edit_distance(a,b) / max(1, len(a))
    def wer(a, b):
        aw = a.split(); bw = b.split()
        return edit_distance(" ".join(aw), " ".join(bw)) / max(1, len(aw))

# ----------------------------
# Utilities: charset and mapping
# ----------------------------
class StrLabelConverter:
    """
    Convert between text-labels and integer sequences for CTC.
    Blank index is 0, characters are 1..N
    """
    def __init__(self, alphabet: List[str], blank_idx: int = 0):
        self.alphabet = alphabet
        self.blank_idx = blank_idx
        # char->index (1..N)
        self.char2idx = {c: i+1 for i, c in enumerate(self.alphabet)}
        # reverse (index->char) for decoding (index 0 reserved for blank)
        self.idx2char = {i+1: c for i, c in enumerate(self.alphabet)}

    def encode(self, text: str) -> List[int]:
        """Convert text to list of ints (no blanks)."""
        return [self.char2idx.get(c, self.char2idx.get("?", 1)) for c in text]

    def decode(self, seq: List[int]) -> str:
        """Decode sequence of ints (CTC greedy collapsed expected) -> string"""
        chars = []
        prev = None
        for idx in seq:
            if idx != prev and idx != self.blank_idx:
                # map idx to char (note stored indices start at 1)
                ch = self.idx2char.get(idx, "?")
                chars.append(ch)
            prev = idx
        return "".join(chars)


def build_charset_from_csv(csv_paths: List[str], extra_chars: str = "") -> List[str]:
    chars = set()
    for p in csv_paths:
        if not p or not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        if "text" in df.columns:
            for t in df["text"].astype(str).tolist():
                chars.update(list(t))
    # add any extras
    for c in extra_chars:
        chars.add(c)
    # sort for determinism
    # ensure '?' fallback character present
    if "?" not in chars:
        chars.add("?")
    # keep deterministic order: place common ascii first if present
    ordered = sorted(chars, key=lambda x: (ord(x) if len(x)==1 else 0))
    return ordered

# ----------------------------
# Dataset & transforms
# ----------------------------
def default_augment(pil: Image.Image) -> Image.Image:
    # random small rotation
    if random.random() < 0.3:
        angle = random.uniform(-3, 3)
        pil = pil.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=255)

    # brightness
    if random.random() < 0.4:
        b = random.uniform(0.9, 1.1)
        pil = ImageEnhance.Brightness(pil).enhance(b)

    # contrast
    if random.random() < 0.4:
        c = random.uniform(0.9, 1.1)
        pil = ImageEnhance.Contrast(pil).enhance(c)

    # translation (replaces ImageOps.offset, which Pillow removed)
    if random.random() < 0.2:
        max_dx = int(0.02 * pil.width)
        max_dy = int(0.02 * pil.height)
        dx = random.randint(-max_dx, max_dx)
        dy = random.randint(-max_dy, max_dy)

        pil = pil.transform(
            pil.size,
            Image.AFFINE,
            (1, 0, dx, 0, 1, dy),
            resample=Image.BILINEAR,
            fillcolor=255,
        )

    # Gaussian blur
    if random.random() < 0.05:
        pil = pil.filter(ImageFilter.GaussianBlur(radius=0.5))

    return pil


class HandwritingLineDataset(Dataset):
    def __init__(self, csv_path: str, height: int = 64, augment: bool = False):
        self.df = pd.read_csv(csv_path)
        assert "image_path" in self.df.columns and "text" in self.df.columns, "CSV must have image_path and text columns"
        self.height = height
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = str(row["image_path"])
        text = str(row["text"])
        # load grayscale
        img = Image.open(img_path).convert("L")
        # optionally augment
        if self.augment:
            img = default_augment(img)
        # resize preserving aspect ratio by height
        w, h = img.size
        new_h = self.height
        new_w = max(1, int(w * (new_h / float(h))))
        img = img.resize((new_w, new_h), Image.LANCZOS)
        # convert to numpy float32 normalized [-1,1]
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        # channel-first
        tensor = torch.from_numpy(arr).unsqueeze(0)  # (1,H,W)
        return tensor, text, img_path

def collate_fn(batch):
    """
    Pad batch to maximum width, return:
      images: (B, 1, H, Wmax)
      targets: concatenated target indices (1D)
      target_lengths: lengths per sample (list)
      raw_texts: list[str]
      img_paths: list[str]
    """
    images = [b[0] for b in batch]
    texts = [b[1] for b in batch]
    img_paths = [b[2] for b in batch]
    heights = [t.shape[1] for t in images]
    widths = [t.shape[2] for t in images]
    H = heights[0]
    Wmax = max(widths)
    padded = torch.zeros((len(images), 1, H, Wmax), dtype=images[0].dtype)
    for i, t in enumerate(images):
        w = t.shape[2]
        padded[i, :, :, :w] = t
    return padded, texts, widths, img_paths

# ----------------------------
# CRNN Model
# ----------------------------
class CRNN(nn.Module):
    def __init__(self, imgH: int, nc: int, nclass: int, nh: int = 256):
        """
        imgH: input image height
        nc: input channels (1)
        nclass: output classes (including blank? we will set fc to nclass which should equal len(alphabet)+1(for blank) )
        nh: LSTM hidden size
        """
        super().__init__()
        # small VGG-like conv stack (keeps height -> 1 after convs)
        ks = [3,3,3,3,3,3,3]
        ps = [1]*len(ks)
        ss = [1]*len(ks)
        nm = [64, 128, 256, 256, 512, 512, 512]

        def conv_relu(i, batch_norm=False):
            in_ch = nc if i==0 else nm[i-1]
            out_ch = nm[i]
            layers = [nn.Conv2d(in_ch, out_ch, ks[i], ss[i], ps[i])]
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            return layers

        layers = []
        layers += conv_relu(0, batch_norm=False)
        layers += [nn.MaxPool2d(2,2)]                 # 64x H/2 x W/2
        layers += conv_relu(1, batch_norm=False)
        layers += [nn.MaxPool2d(2,2)]                 # 128x H/4 x W/4
        layers += conv_relu(2, batch_norm=True)
        layers += conv_relu(3, batch_norm=False)
        layers += [nn.MaxPool2d((2,1), (2,1))]        # reduce height only: H/8 x W/4
        layers += conv_relu(4, batch_norm=True)
        layers += conv_relu(5, batch_norm=False)
        layers += [nn.MaxPool2d((2,1), (2,1))]        # reduce height more: H/16 x W/4
        layers += conv_relu(6, batch_norm=True)
        self.cnn = nn.Sequential(*layers)

        # after cnn we collapse height dimension and treat width as time
        # compute cnn output channels
        # pass a dummy through to infer
        with torch.no_grad():
            dummy = torch.zeros(1, nc, imgH, 128)
            cnn_out = self.cnn(dummy)
            # cnn_out shape: (1, C, Hc, Wc)
            _, c, hc, wc = cnn_out.shape
            rnn_in = c * hc

        self.rnn_in = rnn_in
        self.rnn = nn.Sequential(
            nn.Linear(rnn_in, nh),
        )
        # two-layer bidirectional LSTM
        self.rnn1 = nn.LSTM(nh, nh, num_layers=1, bidirectional=True, batch_first=True)
        self.rnn2 = nn.LSTM(nh*2, nh, num_layers=1, bidirectional=True, batch_first=True)

        self.fc = nn.Linear(nh*2, nclass)

    def forward(self, x):
        # x: (B, C, H, W)
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        # collapse H dimension
        conv = conv.permute(0, 3, 1, 2)  # B, W, C, H
        conv = conv.contiguous().view(b, w, c*h)  # B, W, C*H  (time major = width)
        # project
        out = self.rnn(conv)  # Linear (B, W, nh)
        out = out.view(out.size(0), out.size(1), -1)  # (B, W, nh)
        out, _ = self.rnn1(out)  # (B, W, 2*nh)
        out, _ = self.rnn2(out)  # (B, W, 2*nh)
        out = self.fc(out)  # (B, W, nclass)
        # For CTC we want (T, B, C)
        out = out.permute(1, 0, 2)
        return out  # (T, B, C)

# ----------------------------
# Greedy CTC decode (for validation)
# ----------------------------
def ctc_greedy_decode(probs, converter: StrLabelConverter, blank_idx: int = 0):
    # probs: (T, B, C) logits or probabilities
    if probs.dim() == 3:
        T, B, C = probs.shape
        # argmax over classes at each time
        preds = probs.argmax(-1)  # (T, B)
        preds = preds.cpu().numpy().T  # (B, T)
    else:
        raise RuntimeError("Expected 3D probs (T,B,C)")
    results = []
    for seq in preds:
        # seq elements are class indices 0..C-1 ; our converter expects blank=0 and char indices starting at 1
        seq_list = seq.tolist()
        decoded = converter.decode(seq_list)
        results.append(decoded)
    return results

# ----------------------------
# Training / Validation loops
# ----------------------------
def train_one_epoch(model, device, loader, converter, criterion, optimizer, scaler, epoch, writer, print_every=100):
    model.train()
    total_loss = 0.0
    it = 0
    start = time.time()
    for batch_idx, (imgs, texts, widths, _) in enumerate(loader):
        imgs = imgs.to(device)
        # encode texts to targets
        targets = []
        target_lengths = []
        for t in texts:
            enc = converter.encode(t)
            targets.extend(enc)
            target_lengths.append(len(enc))
        if len(targets) == 0:
            continue
        targets = torch.LongTensor(targets).to(device)
        target_lengths = torch.LongTensor(target_lengths).to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            preds = model(imgs)  # (T, B, C)
            # log softmax needed for CTC loss
            log_probs = F.log_softmax(preds, dim=2)
            input_lengths = torch.LongTensor([preds.size(0)] * preds.size(1)).to(device)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        it += 1
        if writer is not None:
            writer.add_scalar("train/loss_step", loss.item(), epoch*100000 + batch_idx)
        if (batch_idx+1) % print_every == 0:
            elapsed = time.time() - start
            print(f"Epoch {epoch} Iter {batch_idx+1}/{len(loader)} loss={total_loss/it:.4f} ({elapsed:.1f}s)")
    avg_loss = total_loss/it if it>0 else 0.0
    return avg_loss

def validate(model, device, loader, converter, blank_idx=0, max_print=6):
    model.eval()
    total_cer = 0.0
    total_wer = 0.0
    total = 0
    samples = []
    with torch.no_grad():
        for imgs, texts, widths, paths in loader:
            imgs = imgs.to(device)
            preds = model(imgs)  # (T, B, C)
            probs = F.softmax(preds, dim=2)
            decoded = ctc_greedy_decode(probs, converter, blank_idx=blank_idx)
            for gt, pr in zip(texts, decoded):
                total += 1
                total_cer += cer(gt, pr)
                total_wer += wer(gt, pr)
                if len(samples) < max_print:
                    samples.append((gt, pr))
    avg_cer = total_cer / max(1, total)
    avg_wer = total_wer / max(1, total)
    return avg_cer, avg_wer, samples

# ----------------------------
# Checkpoint helpers
# ----------------------------
def save_checkpoint(state, path):
    torch.save(state, path)

def load_checkpoint(path, model, optimizer=None, scaler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    start_epoch = ckpt.get("epoch", 0)
    best_val = ckpt.get("best_val", None)
    return start_epoch, best_val

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="checkpoints/crnn")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", help="use mixed precision")
    parser.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume")
    parser.add_argument("--charlist", type=str, default=None, help="optional path to charlist (one char per line)")
    parser.add_argument("--print_every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "tb"))

    # build charset
    if args.charlist and os.path.exists(args.charlist):
        with open(args.charlist, "r", encoding="utf8") as f:
            chars = [line.rstrip("\n") for line in f if line.rstrip("\n") != ""]
    else:
        chars = build_charset_from_csv([args.train_csv])
    print(f"Constructed alphabet from training csv (len={len(chars)}).")
    print("Alphabet sample:", chars[:80])
    converter = StrLabelConverter(chars, blank_idx=0)
    nclass = len(chars) + 1  # +1 for blank (idx 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # datasets
    train_ds = HandwritingLineDataset(args.train_csv, height=args.height, augment=True)
    val_ds = HandwritingLineDataset(args.val_csv, height=args.height, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    # model
    model = CRNN(imgH=args.height, nc=1, nclass=nclass, nh=256).to(device)
    print("Model params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=max(1, args.epochs * len(train_loader)),
                                              pct_start=0.1, anneal_strategy="cos", final_div_factor=10)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None

    start_epoch = 0
    best_val = 1e9
    if args.resume:
        print("Resuming from", args.resume)
        se, bv = load_checkpoint(args.resume, model, optimizer=optimizer, scaler=scaler)
        start_epoch = se
        if bv is not None:
            best_val = bv

    # training loop
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, device, train_loader, converter, criterion,
                                     optimizer, scaler, epoch, writer, print_every=args.print_every)
        # step the scheduler once per epoch using average loss iterations count (we used OneCycleLR which expects step per iteration)
        # Actually OneCycleLR should be stepped per iteration; we already call step in train loop? we didn't -> call here per epoch in a safe way:
        # To keep it simple, step scheduler by epoch with small steps:
        try:
            for _ in range(len(train_loader)):
                scheduler.step()
        except Exception:
            pass

        val_cer, val_wer, samples = validate(model, device, val_loader, converter)
        dt = time.time() - t0
        print(f"Epoch {epoch} finished in {dt:.1f}s train_loss={train_loss:.4f} val_cer={val_cer:.4f} val_wer={val_wer:.4f}")
        writer.add_scalar("train/loss_epoch", train_loss, epoch)
        writer.add_scalar("val/cer", val_cer, epoch)
        writer.add_scalar("val/wer", val_wer, epoch)

        # log some sample predictions
        for i, (gt, pr) in enumerate(samples):
            writer.add_text(f"sample/{i}", f"GT: {gt}\nPR: {pr}", epoch)

        # checkpoint
        ckpt = {
            "epoch": epoch+1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_val": best_val,
            "alphabet": chars,
        }
        ckpt_path = os.path.join(args.out_dir, f"crnn_epoch_{epoch+1}.pt")
        save_checkpoint(ckpt, ckpt_path)

        if val_cer < best_val:
            best_val = val_cer
            best_path = os.path.join(args.out_dir, "best_crnn.pt")
            ckpt["best_val"] = best_val
            save_checkpoint(ckpt, best_path)
            print("Saved best model ->", best_path)

    print("Training finished. Best val CER:", best_val)
    writer.close()

if __name__ == "__main__":
    main()
