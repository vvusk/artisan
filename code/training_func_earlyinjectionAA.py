import copy
import gc
import os
import warnings

import numpy as np
import pandas as pd
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

warnings.filterwarnings("ignore")

from data_preprocessing import prog_labellist
from Dataloader import train_dataset, TRAIN_INDICES
from Model import GPTS_EarlyInjectionAA

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# CONFIG
# -------------------------
NUM_ARCHETYPES = 18      # MUST match compute_aa_weights.py and Model.GPTS
MAX_EPOCHS = 100
BATCH_SIZE = 5
N_SPLITS = 5
RANDOM_STATE = 42

# Early stopping
PATIENCE = 10
MIN_DELTA = 0.0

SAVE_DIR = "./saved_models_GPTS_EarlyInjectionAA"
PER_FOLD_CSV = "cv_metrics_EarlyInjectionAA.csv"
SUMMARY_CSV = "cv_metrics_summary_EarlyInjectionAA.csv"

os.makedirs(SAVE_DIR, exist_ok=True)

criterion = nn.BCEWithLogitsLoss()


def compute_metrics(y_true, y_pred, y_prob):
    """
    Compute binary classification metrics.

    AUC is set to np.nan when it cannot be computed, for example
    when a validation fold contains only one class.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_measure": f1_score(y_true, y_pred, zero_division=0),
    }

    try:
        metrics["auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["auc"] = np.nan

    return metrics


# -------------------------
# LABELS FOR TRAIN DATASET
# -------------------------
labels_np = np.array(prog_labellist)
labels_train = labels_np[TRAIN_INDICES]   # labels corresponding to train_dataset
n_train = len(train_dataset)

print("Train dataset size:", n_train)
print("Train class counts (0=stable, 1=progressor):",
      (labels_train == 0).sum(), (labels_train == 1).sum())

# -------------------------
# 5-FOLD STRATIFIED CV ON TRAIN DATASET
# -------------------------
skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

fold_metrics = []
fold_best_states = []

fold_idx = 1
for train_idx_rel, val_idx_rel in skf.split(
    np.zeros(len(labels_train)), labels_train
):
    print("\n========================")
    print(f"Fold {fold_idx}/{N_SPLITS}")
    print("========================")

    # train_idx_rel / val_idx_rel are indices relative to train_dataset
    train_subset = Subset(train_dataset, train_idx_rel)
    val_subset = Subset(train_dataset, val_idx_rel)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Fold {fold_idx} | Train size: {len(train_subset)} | Val size: {len(val_subset)}")

    # New model & optimizer for this fold
    model = GPTS_EarlyInjectionAA(num_archetypes=NUM_ARCHETYPES).to(device)
    optimizer = SGD(model.parameters(), lr=0.1)

    best_val_loss = float("inf")
    best_state_dict = None
    best_epoch = 0
    best_metrics = None
    epochs_no_improve = 0

    # ---- TRAINING LOOP FOR THIS FOLD ----
    for epoch in range(MAX_EPOCHS):
        # ----------------- TRAIN -----------------
        model.train()
        train_losses = []
        train_true = []
        train_pred = []

        for data, aa_w, label in train_loader:
            data = data.to(device).float()          # (B,3,3,9,9)
            aa_w = aa_w.to(device).float()          # (B,K)
            label = label.unsqueeze(1).to(device)   # (B,1)

            optimizer.zero_grad()
            logits = model(data, aa_w)              # (B,1)
            loss = criterion(logits, label.float())
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            # logits >= 0 is equivalent to sigmoid(logits) >= 0.5
            preds_binary = (logits >= 0).detach().cpu().numpy().astype(int)
            labels_np_b = label.detach().cpu().numpy().astype(int)
            train_pred.extend(preds_binary.flatten())
            train_true.extend(labels_np_b.flatten())

        train_acc = accuracy_score(train_true, train_pred)
        train_loss = float(np.mean(train_losses))

        # ----------------- VALIDATION -----------------
        model.eval()
        val_losses = []
        val_true = []
        val_pred = []
        val_prob = []

        with torch.no_grad():
            for data, aa_w, label in val_loader:
                data = data.to(device).float()
                aa_w = aa_w.to(device).float()
                label = label.unsqueeze(1).to(device)

                logits = model(data, aa_w)
                loss = criterion(logits, label.float())

                val_losses.append(loss.item())

                probs = torch.sigmoid(logits).cpu().numpy()
                preds_binary = (logits >= 0).cpu().numpy().astype(int)
                labels_np_b = label.cpu().numpy().astype(int)

                val_prob.extend(probs.flatten())
                val_pred.extend(preds_binary.flatten())
                val_true.extend(labels_np_b.flatten())

        val_loss = float(np.mean(val_losses))
        val_metrics = compute_metrics(val_true, val_pred, val_prob)

        print(f"Fold {fold_idx} | Epoch {epoch+1}/{MAX_EPOCHS} "
              f"| train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f} "
              f"| val_loss: {val_loss:.4f}, val_acc: {val_metrics['accuracy']:.4f}, "
              f"val_precision: {val_metrics['precision']:.4f}, "
              f"val_recall: {val_metrics['recall']:.4f}, "
              f"val_f1: {val_metrics['f1_measure']:.4f}, "
              f"val_auc: {val_metrics['auc']:.4f}")

        # ---- EARLY STOPPING on val_loss ----
        if val_loss + MIN_DELTA < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_metrics = val_metrics.copy()
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1} for fold {fold_idx} "
                      f"(no val_loss improvement in {PATIENCE} epochs).")
                break

        gc.collect()
        torch.cuda.empty_cache()

    # Save best model for this fold
    if best_state_dict is not None:
        fold_model_path = os.path.join(SAVE_DIR, f"best_model_fold_{fold_idx}.pth")
        torch.save(best_state_dict, fold_model_path)
        print(f"Saved best model for fold {fold_idx} with "
              f"val_loss={best_val_loss:.4f}, val_acc={best_metrics['accuracy']:.4f}")

    fold_result = {
        "fold": fold_idx,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "accuracy": best_metrics["accuracy"],
        "precision": best_metrics["precision"],
        "recall": best_metrics["recall"],
        "f1_measure": best_metrics["f1_measure"],
        "auc": best_metrics["auc"],
    }

    fold_metrics.append(fold_result)
    fold_best_states.append(best_state_dict)
    fold_idx += 1

# -------------------------
# SUMMARY OF CV
# -------------------------
metrics_df = pd.DataFrame(fold_metrics)
metrics_df.to_csv(PER_FOLD_CSV, index=False)

summary_rows = []
for metric_name in ["accuracy", "precision", "recall", "f1_measure", "auc"]:
    summary_rows.append({
        "metric": metric_name,
        "mean": metrics_df[metric_name].mean(skipna=True),
        "std": metrics_df[metric_name].std(skipna=True),
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_CSV, index=False)

print("\n========================")
print("5-FOLD CV SUMMARY")
print("========================")
for _, row in metrics_df.iterrows():
    print(f"Fold {int(row['fold'])} | "
          f"accuracy: {row['accuracy']:.4f}, "
          f"precision: {row['precision']:.4f}, "
          f"recall: {row['recall']:.4f}, "
          f"F1-measure: {row['f1_measure']:.4f}, "
          f"AUC: {row['auc']:.4f}")

print("\nMean ± Std")
for _, row in summary_df.iterrows():
    print(f"{row['metric']}: {row['mean']:.4f} ± {row['std']:.4f}")

print(f"\nSaved per-fold metrics to: {PER_FOLD_CSV}")
print(f"Saved summary metrics to: {SUMMARY_CSV}")

# Pick the fold with highest validation accuracy to evaluate on TEST
best_fold_index = int(metrics_df["accuracy"].idxmax())
best_state_for_test = fold_best_states[best_fold_index]
best_model_for_test_path = os.path.join(SAVE_DIR, "best_model_for_test.pth")
torch.save(best_state_for_test, best_model_for_test_path)

print(f"\nBest fold for test is fold {best_fold_index+1} "
      f"with val_acc={metrics_df.loc[best_fold_index, 'accuracy']:.4f}")
print(f"Saved as {best_model_for_test_path}")

print("Training + cross-validation complete!")
