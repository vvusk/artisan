import torch
import torch.nn as nn

from timesformer_pytorch import TimeSformer
from TimeSformer_with_AA import TimeSformer_with_AA


# 1) Baseline: No AA
class GPTS_NoAA(nn.Module):
    def __init__(self):
        super(GPTS_NoAA, self).__init__()
        self.model = TimeSformer(
            dim=512,
            image_size=9,
            patch_size=3,
            num_frames=3,
            num_classes=1,
            depth=12,
            heads=8,
            dim_head=64,
            attn_dropout=0,
            ff_dropout=0
        )

    def forward(self, x):
        return self.model(x)


# 2) Late Fusion AA: concat after TimeSformer
class GPTS_LateFusionAA(nn.Module):
    def __init__(self, num_archetypes=18, aa_hidden_dim=64):
        super(GPTS_LateFusionAA, self).__init__()

        self.model = TimeSformer(
            dim=512,
            image_size=9,
            patch_size=3,
            num_frames=3,
            num_classes=1,
            depth=12,
            heads=8,
            dim_head=64,
            attn_dropout=0,
            ff_dropout=0
        )

        self.aa_mlp = nn.Sequential(
            nn.Linear(num_archetypes, aa_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        self.classifier = nn.Linear(1 + aa_hidden_dim, 1)

    def forward(self, x, aa_w):
        ts_out = self.model(x)              # (B,1)
        aa_feat = self.aa_mlp(aa_w)         # (B,aa_hidden_dim)
        combined = torch.cat([ts_out, aa_feat], dim=1)
        out = self.classifier(combined)     # (B,1)
        return out


# 3) Early Injection AA: AA enters transformer input
class GPTS_EarlyInjectionAA(nn.Module):
    def __init__(self, num_archetypes=18):
        super(GPTS_EarlyInjectionAA, self).__init__()

        self.model = TimeSformer_with_AA(
            dim=512,
            image_size=9,
            patch_size=3,
            num_frames=3,
            num_classes=1,
            num_archetypes=num_archetypes,
            depth=12,
            heads=8,
            dim_head=64,
            attn_dropout=0,
            ff_dropout=0
        )

    def forward(self, x, aa_w):
        return self.model(x, aa_w)