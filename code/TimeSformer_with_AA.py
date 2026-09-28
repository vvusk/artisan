import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

from timesformer_pytorch.rotary import apply_rot_emb, AxialRotaryEmbedding, RotaryEmbedding


def exists(val):
    return val is not None


NUM_SPECIAL_TOKENS = 2   # CLS + AA


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


def shift(t, amt):
    if amt == 0:
        return t
    return F.pad(t, (0, 0, 0, 0, amt, -amt))


class PreTokenShift(nn.Module):
    def __init__(self, frames, fn):
        super().__init__()
        self.frames = frames
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        f, dim = self.frames, x.shape[-1]

        # separate special tokens from patch tokens
        special_x, x = x[:, :NUM_SPECIAL_TOKENS], x[:, NUM_SPECIAL_TOKENS:]
        x = rearrange(x, 'b (f n) d -> b f n d', f=f)

        dim_chunk = (dim // 3)
        chunks = x.split(dim_chunk, dim=-1)
        chunks_to_shift, rest = chunks[:3], chunks[3:]
        shifted_chunks = tuple(
            map(lambda args: shift(*args), zip(chunks_to_shift, (-1, 0, 1)))
        )
        x = torch.cat((*shifted_chunks, *rest), dim=-1)

        x = rearrange(x, 'b f n d -> b (f n) d')
        x = torch.cat((special_x, x), dim=1)
        return self.fn(x, *args, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


def attn(q, k, v, mask=None):
    sim = einsum('b i d, b j d -> b i j', q, k)

    if exists(mask):
        max_neg_value = -torch.finfo(sim.dtype).max
        sim.masked_fill_(~mask, max_neg_value)

    attn_map = sim.softmax(dim=-1)
    out = einsum('b i j, b j d -> b i d', attn_map, v)
    return out


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        dropout=0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        einops_from,
        einops_to,
        mask=None,
        cls_mask=None,
        rot_emb=None,
        **einops_dims
    ):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        q = q * self.scale

        # separate special tokens (CLS + AA) from patch tokens
        (special_q, q_), (special_k, k_), (special_v, v_) = map(
            lambda t: (t[:, :NUM_SPECIAL_TOKENS], t[:, NUM_SPECIAL_TOKENS:]),
            (q, k, v)
        )

        # let special tokens attend to everything
        special_out = attn(special_q, k, v, mask=cls_mask)

        # rearrange only patch tokens across time or space
        q_, k_, v_ = map(
            lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims),
            (q_, k_, v_)
        )

        if exists(rot_emb):
            q_, k_ = apply_rot_emb(q_, k_, rot_emb)

        # repeat special tokens across time or space groups and prepend them
        r = q_.shape[0] // special_k.shape[0]
        special_k = repeat(special_k, 'b s d -> (b r) s d', r=r)
        special_v = repeat(special_v, 'b s d -> (b r) s d', r=r)

        k_ = torch.cat((special_k, k_), dim=1)
        v_ = torch.cat((special_v, v_), dim=1)

        out = attn(q_, k_, v_, mask=mask)

        # merge back time or space
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)

        # concat back special outputs
        out = torch.cat((special_out, out), dim=1)

        # merge heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        return self.to_out(out)


class TimeSformer_with_AA(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_frames,
        num_classes,
        num_archetypes=18,
        image_size=224,
        patch_size=16,
        channels=3,
        depth=12,
        heads=8,
        dim_head=64,
        attn_dropout=0.,
        ff_dropout=0.,
        rotary_emb=True,
        shift_tokens=False
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_size // patch_size) ** 2
        num_positions = num_frames * num_patches
        patch_dim = channels * patch_size ** 2

        self.heads = heads
        self.patch_size = patch_size
        self.to_patch_embedding = nn.Linear(patch_dim, dim)

        self.cls_token = nn.Parameter(torch.randn(1, dim))
        self.aa_proj = nn.Linear(num_archetypes, dim)

        self.use_rotary_emb = rotary_emb
        if rotary_emb:
            self.frame_rot_emb = RotaryEmbedding(dim_head)
            self.image_rot_emb = AxialRotaryEmbedding(dim_head)
        else:
            self.pos_emb = nn.Embedding(num_positions + NUM_SPECIAL_TOKENS, dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            ff = FeedForward(dim, dropout=ff_dropout)
            time_attn = Attention(dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)
            spatial_attn = Attention(dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)

            if shift_tokens:
                time_attn, spatial_attn, ff = map(
                    lambda t: PreTokenShift(num_frames, t),
                    (time_attn, spatial_attn, ff)
                )

            time_attn, spatial_attn, ff = map(
                lambda t: PreNorm(dim, t),
                (time_attn, spatial_attn, ff)
            )

            self.layers.append(nn.ModuleList([time_attn, spatial_attn, ff]))

        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, video, aa_w, mask=None):
        b, f, _, h, w, *_, device, p = *video.shape, video.device, self.patch_size
        assert h % p == 0 and w % p == 0, f'height {h} and width {w} of video must be divisible by the patch size {p}'

        hp, wp = (h // p), (w // p)
        n = hp * wp

        # video to patch embeddings
        video = rearrange(
            video,
            'b f c (h p1) (w p2) -> b (f h w) (p1 p2 c)',
            p1=p,
            p2=p
        )
        tokens = self.to_patch_embedding(video)   # (B, f*n, dim)

        # special tokens
        cls_token = repeat(self.cls_token, 'n d -> b n d', b=b)   # (B,1,D)
        aa_token = self.aa_proj(aa_w).unsqueeze(1)                # (B,1,D)

        # [CLS | AA | patches]
        x = torch.cat((cls_token, aa_token, tokens), dim=1)

        frame_pos_emb = None
        image_pos_emb = None
        if not self.use_rotary_emb:
            x += self.pos_emb(torch.arange(x.shape[1], device=device))
        else:
            frame_pos_emb = self.frame_rot_emb(f, device=device)
            image_pos_emb = self.image_rot_emb(hp, wp, device=device)

        frame_mask = None
        cls_attn_mask = None
        if exists(mask):
            # frame mask is used only for patch-token reshaped attention
            frame_mask = repeat(mask, 'b f -> (b h n) () f', n=n, h=self.heads)

            # special tokens can attend to all valid patch tokens + both special tokens
            cls_attn_mask = repeat(mask, 'b f -> (b h) () (f n)', n=n, h=self.heads)
            cls_attn_mask = F.pad(cls_attn_mask, (NUM_SPECIAL_TOKENS, 0), value=True)

        for (time_attn, spatial_attn, ff) in self.layers:
            x = time_attn(
                x,
                'b (f n) d',
                '(b n) f d',
                n=n,
                mask=frame_mask,
                cls_mask=cls_attn_mask,
                rot_emb=frame_pos_emb
            ) + x

            x = spatial_attn(
                x,
                'b (f n) d',
                '(b f) n d',
                f=f,
                cls_mask=cls_attn_mask,
                rot_emb=image_pos_emb
            ) + x

            x = ff(x) + x

        cls_token = x[:, 0]
        return self.to_out(cls_token)