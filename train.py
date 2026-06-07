"""
train.py — Fine-tune ViT-B/16 on APTOS 2019 Diabetic Retinopathy Dataset
==========================================================================

Usage:
    python train.py --data_dir ./aptos_data --epochs 20 --batch_size 32

Dataset structure expected:
    aptos_data/
        train_images/     ← .png fundus images
        train.csv         ← id_code, diagnosis (0-4)
        test_images/
        test.csv
"""

import os
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import timm
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy

from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score


# ── Dataset ───────────────────────────────────────────────────────────────────
class APTOSDataset(Dataset):
    """APTOS 2019 Fundus Image Dataset."""

    def __init__(self, df: pd.DataFrame, img_dir: str, transform=None, is_train=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / f"{row['id_code']}.png"
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = int(row["diagnosis"])
        return image, label


# ── Transforms ────────────────────────────────────────────────────────────────
def get_transforms(img_size=224, is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(num_classes=5, pretrained=True):
    """ViT-B/16 from timm with custom head."""
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model


# ── Metrics ───────────────────────────────────────────────────────────────────
def quadratic_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


# ── Training loop ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(imgs)
            loss = criterion(logits, labels)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        running_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader)
    kappa = quadratic_kappa(all_labels, all_preds)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return avg_loss, kappa, acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        running_loss += loss.item()

        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader)
    kappa = quadratic_kappa(all_labels, all_preds)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return avg_loss, kappa, acc


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load CSV
    df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    print(f"Total samples: {len(df)}")

    # K-Fold cross-validation
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["diagnosis"])):
        print(f"\n{'='*50}")
        print(f"FOLD {fold+1}/{args.folds}")
        print(f"{'='*50}")

        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        train_ds = APTOSDataset(train_df, os.path.join(args.data_dir, "train_images"),
                                get_transforms(is_train=True))
        val_ds = APTOSDataset(val_df, os.path.join(args.data_dir, "train_images"),
                              get_transforms(is_train=False))

        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2,
                                shuffle=False, num_workers=4, pin_memory=True)

        model = build_model().to(device)
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

        best_kappa = -1
        best_path = f"best_fold{fold+1}.pth"

        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_kappa, tr_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler)
            val_loss, val_kappa, val_acc = validate(model, val_loader, criterion, device)
            scheduler.step()

            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Train Loss {tr_loss:.4f} Kappa {tr_kappa:.4f} Acc {tr_acc:.3f} | "
                  f"Val Loss {val_loss:.4f} Kappa {val_kappa:.4f} Acc {val_acc:.3f}")

            if val_kappa > best_kappa:
                best_kappa = val_kappa
                torch.save(model.state_dict(), best_path)
                print(f"  ✓ Saved best model (kappa={best_kappa:.4f})")

        fold_results.append(best_kappa)
        print(f"Best kappa fold {fold+1}: {best_kappa:.4f}")

    print(f"\n{'='*50}")
    print(f"CV Quadratic Kappa: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViT on APTOS 2019")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    main(args)
