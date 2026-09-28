# compute_aa_weights.py
import numpy as np
from data_preprocessing import datalist
from AA_Original_init import ArchetypalAnalysis1  # your AA class

# Choose number of archetypes (tune later: 8, 10, 12, etc.)
N_ARCHETYPES = 18

# Build feature matrix X: one row per sample
# Here we use the FIRST frame, FIRST channel (since all 3 channels are identical copies)
X = []
for video in datalist:
    # video shape ~ (T=3, C=3, H=9, W=9)
    first_frame = video[0, 0, :, :]       # (9, 9)
    X.append(first_frame.flatten())       # (81,)

X = np.stack(X, axis=0)                   # (N_samples, 81)

# Fit Archetypal Analysis
aa = ArchetypalAnalysis1(
    n_archetypes=N_ARCHETYPES,
    random_state=0
)
aa.fit(X)

# aa.alfa is typically shape (n_samples, n_archetypes) or (n_archetypes, n_samples)
alpha = aa.alfa
# If your alpha shape is (K, N), transpose it:
if alpha.shape[0] == N_ARCHETYPES and alpha.shape[1] == X.shape[0]:
    alpha = alpha.T   # (N_samples, N_ARCHETYPES)

print("Alpha shape (n_samples, n_archetypes):", alpha.shape)

# Save to disk
np.save("aa_weights.npy", alpha)
print("Saved archetype weights to aa_weights.npy")
