import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

from data_preprocessing import datalist, prog_labellist, archetype_weights

class UWHVFDataset(Dataset):
    def __init__(self, data, label, aa_weights=None, transform=None):
        self.data = data
        self.label = label
        self.aa_weights = aa_weights
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]

        if self.aa_weights is None:
            raise RuntimeError(
                "AA weights are None. Make sure compute_aa_weights.py "
                "and data_preprocessing.py have been run correctly."
            )
        aa_w = self.aa_weights[index].astype(np.float32)

        if self.transform is not None:
            x = self.transform(x)

        return x, aa_w, y

# -------------------------
# BUILD FULL DATASET
# -------------------------
full_dataset = UWHVFDataset(
    data=datalist,
    label=prog_labellist,
    aa_weights=archetype_weights,
    transform=None
)

labels_np = np.array(prog_labellist)
n_samples = len(full_dataset)
all_indices = np.arange(n_samples)

# -------------------------
# SINGLE STRATIFIED SPLIT: TRAIN (80%) / TEST (20%)
# -------------------------
train_indices, test_indices = train_test_split(
    all_indices,
    test_size=0.2,
    stratify=labels_np,
    random_state=42
)

train_dataset = Subset(full_dataset, train_indices)
test_dataset  = Subset(full_dataset, test_indices)

print(f"Total samples: {n_samples}")
print(f"Train size   : {len(train_dataset)}")
print(f"Test size    : {len(test_dataset)}")

# You can still expose loaders for convenience
BATCH_SIZE = 5

dataloaders = {
    "train": DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True),
    "test":  DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False),
}

# Optionally expose the indices so training_func.py can get labels for CV
TRAIN_INDICES = train_indices
TEST_INDICES  = test_indices
