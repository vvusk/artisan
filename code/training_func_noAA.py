import os
import csv
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import warnings
import gc

warnings.filterwarnings("ignore")

from data_preprocessing import prog_labellist
from Dataloader import train_dataset, TRAIN_INDICES
from Model import GPTS_NoAA   # <-- you need this in Model.py

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# CONFIG
# -------------------------
MAX_EPOCHS = 100
BATCH_SIZE = 5
N_SPLITS = 5
RANDOM_STATE = 42

# Early stopping
PATIENCE = 10
MIN_DELTA = 0.0

criterion = nn.BCEWithLogitsLoss()

# Output directory
MODEL_DIR = "./saved_models_GPTS_noAA"
os.makedirs(MODEL_DIR, exist_ok=True)


def compute_binary_metrics(y_true, y_pred, y_score):
    """
    Compute binary classification metrics.

    y_score should be the predicted probability/logit score for the positive class.
    AUC is set to np.nan if only one class is present in y_true.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_score = np.asarray(y_score).astype(float)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    try:
        metrics["auc"] = roc_auc_score(y_true, y_score)
    except ValueError:
        metrics["auc"] = np.nan

    return metrics

# Labels for CV splitting
labels_np = np.array(prog_labellist)
labels_train = labels_np[TRAIN_INDICES]

print("Train dataset size:", len(train_dataset))
print("Train class counts (0=stable, 1=progressor):",
      (labels_train == 0).sum(), (labels_train == 1).sum())

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

fold_best_val_accs = []
fold_best_metrics = []
fold_best_states = []

fold_idx = 1
for train_idx_rel, val_idx_rel in skf.split(np.zeros(len(labels_train)), labels_train):
    print("\n========================")
    print(f"BASELINE (No AA) | Fold {fold_idx}/{N_SPLITS}")
    print("========================")

    train_subset = Subset(train_dataset, train_idx_rel)
    val_subset   = Subset(train_dataset, val_idx_rel)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE, shuffle=False)

    print(f"Fold {fold_idx} | Train size: {len(train_subset)} | Val size: {len(val_subset)}")
    
    model = GPTS_NoAA().to(device)
    optimizer = SGD(model.parameters(), lr=0.1)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_metrics = None
    best_state_dict = None
    epochs_no_improve = 0

    for epoch in range(MAX_EPOCHS):
        # ---- TRAIN ----
        model.train()
        train_losses, train_true, train_pred, train_score = [], [], [], []

        for data, aa_w, label in train_loader:
            data  = data.to(device).float()
            label = label.unsqueeze(1).to(device)

            optimizer.zero_grad()
            logits = model(data)                 # <-- No AA
            loss = criterion(logits, label.float())
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            probs = torch.sigmoid(logits)
            preds_binary = (probs >= 0.5).detach().cpu().numpy().astype(int)
            train_pred.extend(preds_binary.flatten())
            train_score.extend(probs.detach().cpu().numpy().flatten())
            train_true.extend(label.detach().cpu().numpy().astype(int).flatten())

        train_loss = float(np.mean(train_losses))
        train_metrics = compute_binary_metrics(train_true, train_pred, train_score)
        train_acc = train_metrics["accuracy"]

        # ---- VAL ----
        model.eval()
        val_losses, val_true, val_pred, val_score = [], [], [], []

        with torch.no_grad():
            for data, aa_w, label in val_loader:
                data  = data.to(device).float()
                label = label.unsqueeze(1).to(device)

                logits = model(data)             # <-- No AA
                loss = criterion(logits, label.float())
                val_losses.append(loss.item())

                probs = torch.sigmoid(logits)
                preds_binary = (probs >= 0.5).cpu().numpy().astype(int)
                val_pred.extend(preds_binary.flatten())
                val_score.extend(probs.cpu().numpy().flatten())
                val_true.extend(label.cpu().numpy().astype(int).flatten())

        val_loss = float(np.mean(val_losses))
        val_metrics = compute_binary_metrics(val_true, val_pred, val_score)
        val_acc = val_metrics["accuracy"]

        print(f"Fold {fold_idx} | Epoch {epoch+1}/{MAX_EPOCHS} "
              f"| train_loss: {train_loss:.4f}, train_acc: {train_metrics['accuracy']:.4f}, "
              f"train_precision: {train_metrics['precision']:.4f}, train_recall: {train_metrics['recall']:.4f}, "
              f"train_f1: {train_metrics['f1']:.4f}, train_auc: {train_metrics['auc']:.4f} "
              f"| val_loss: {val_loss:.4f}, val_acc: {val_metrics['accuracy']:.4f}, "
              f"val_precision: {val_metrics['precision']:.4f}, val_recall: {val_metrics['recall']:.4f}, "
              f"val_f1: {val_metrics['f1']:.4f}, val_auc: {val_metrics['auc']:.4f}")

        # Early stopping on val_loss
        if val_loss + MIN_DELTA < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_metrics = val_metrics.copy()
            best_metrics["loss"] = best_val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1} (fold {fold_idx})")
                break

        gc.collect()
        torch.cuda.empty_cache()

    # Save per-fold best model
    torch.save(best_state_dict, os.path.join(MODEL_DIR, f'best_model_fold_{fold_idx}.pth'))

    fold_best_val_accs.append(best_val_acc)
    fold_best_metrics.append(best_metrics)
    fold_best_states.append(best_state_dict)
    fold_idx += 1

# Summary
metric_names = ["accuracy", "precision", "recall", "f1", "auc"]
metrics_array = {
    metric: np.array([fold_metrics[metric] for fold_metrics in fold_best_metrics], dtype=float)
    for metric in metric_names
}

print("\n========================")
print("BASELINE (No AA) CV SUMMARY")
print("========================")
print("Best validation metrics for each fold are taken from the epoch with the lowest validation loss.")

for i, fold_metrics in enumerate(fold_best_metrics, start=1):
    print(f"Fold {i} | "
          f"loss: {fold_metrics['loss']:.4f}, "
          f"accuracy: {fold_metrics['accuracy']:.4f}, "
          f"precision: {fold_metrics['precision']:.4f}, "
          f"recall: {fold_metrics['recall']:.4f}, "
          f"F1-measure: {fold_metrics['f1']:.4f}, "
          f"AUC: {fold_metrics['auc']:.4f}")

print("\nAverage and standard deviation across folds:")
for metric in metric_names:
    print(f"{metric}: mean = {np.nanmean(metrics_array[metric]):.4f}, "
          f"std = {np.nanstd(metrics_array[metric]):.4f}")

# Save CV metrics to CSV
metrics_csv_path = os.path.join(MODEL_DIR, "cv_metrics_noAA.csv")
with open(metrics_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["fold", "loss", "accuracy", "precision", "recall", "F1-measure", "AUC"])
    for i, fold_metrics in enumerate(fold_best_metrics, start=1):
        writer.writerow([
            i,
            fold_metrics["loss"],
            fold_metrics["accuracy"],
            fold_metrics["precision"],
            fold_metrics["recall"],
            fold_metrics["f1"],
            fold_metrics["auc"],
        ])

best_fold_index = int(np.nanargmax(metrics_array["accuracy"]))
best_state_for_test = fold_best_states[best_fold_index]
best_model_for_test_path = os.path.join(MODEL_DIR, "best_model_for_test.pth")
torch.save(best_state_for_test, best_model_for_test_path)
print(f"\nSaved CV metrics -> {metrics_csv_path}")
print(f"Saved baseline best model for test from fold {best_fold_index+1} -> "
      f"{best_model_for_test_path}")
