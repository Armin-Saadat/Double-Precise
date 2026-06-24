import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import logging

# each patch one action
ACTION_2_PATCH_IDX = {0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4], 6: [5], 7: [6], 8: [7],
                      9: [8], 10: [9], 11: [10], 12: [11], 13: [12], 14: [13], 15: [14], 16: [15]}

# # two center columns --> action 2, two side columns --> action 1
# ACTION_2_PATCH_IDX = {0: [], 1: [0, 3, 4, 7, 8, 11, 12, 15], 2: [1, 2, 5, 6, 9, 10, 13, 14]}

# # each column one action
# ACTION_2_PATCH_IDX = {0: [], 1: [0, 4, 8, 12], 2: [1, 5, 9, 13], 3: [2, 6, 10, 14], 4: [3, 7, 11, 15]}


############### Making ViT compatible with RL pipeline and MNIST dataset ###################
class MnistViT:
    def __init__(self, logger: logging.Logger):
        self.patch_size = 7
        self.n_patches = 16
        self.ViT = ViT(
            image_height=28,
            image_width=28,
            n_channels=1,
            patch_size=self.patch_size,
            n_classes=10,
            d_model=64,
            n_layers=6,
            n_heads=8,
            d_mlp=256,
            d_head=8,
            pool='cls',
            dropout=0.1,
            emb_dropout=0.1)

        self.logger = logger

    def validate(self, data: datasets.MNIST, phase: str = 'validation'):
        pass

    def predict(self, state: torch.tensor, action_history: list):
        """
            state: torch.tensor, shape: [N_patches, patch_embeddings]
            action_history: list
        """
        assert 0 not in action_history, f"0 is a termination action, so the process should have been terminated."

        # predict 0 when no data is acquired
        if not action_history:
            return 0

        observed_patches_idx = set([idx for a in action_history for idx in ACTION_2_PATCH_IDX[a]])
        masked_patches_idx = list(set(range(self.n_patches)) - observed_patches_idx)

        with torch.no_grad():
            patch_mask = torch.zeros((1, self.n_patches))
            patch_mask[:, masked_patches_idx] = 1
            pred_labels, attn = self.ViT(images=state.unsqueeze(0), patch_mask=patch_mask)
            _, y_pred = torch.max(pred_labels, dim=1)

        return y_pred.squeeze(0).item()


############### Code for General ViT ###################

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, attn_mask=None, **kwargs):
        return self.fn(self.norm(x), attn_mask=attn_mask, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, attn_mask=None):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, d_head, dropout=0.):
        super().__init__()
        inner_dim = d_head * n_heads
        project_out = not (n_heads == 1 and d_head == d_model)

        self.n_heads = n_heads
        self.scale = d_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(d_model, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, d_model),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, attn_mask=None):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.n_heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if attn_mask is not None:
            # attn_mask shape is expected to be (B, 1, N, N) or broadcastable to (B, h, N, N)
            dots = dots.masked_fill(attn_mask == 0, float('-inf'))

        attn = self.attend(dots)
        out = torch.matmul(self.dropout(attn), v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn


class Transformer(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, d_head, d_mlp, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(nn.ModuleList([
                PreNorm(d_model, Attention(d_model, n_heads, d_head, dropout=dropout)),
                PreNorm(d_model, FeedForward(d_model, d_mlp, dropout=dropout))
            ]))

    def forward(self, x, attn_mask=None):
        for attn, ff in self.layers:
            attn_out, attn_w = attn(x, attn_mask=attn_mask)
            x = attn_out + x
            x = ff(x) + x
        return x, attn_w


class ViT(nn.Module):
    def __init__(self, *, image_height, image_width, n_channels, patch_size, n_classes, d_model, n_layers, n_heads,
                 d_mlp, d_head=64,
                 pool='cls', dropout=0., emb_dropout=0.):
        super().__init__()
        patch_height, patch_width = patch_size, patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        n_patches = (image_height // patch_height) * (image_width // patch_width)
        d_patch = n_channels * patch_height * patch_width

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(d_patch),
            nn.Linear(d_patch, d_model),
            nn.LayerNorm(d_model),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches + 1, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(d_model, n_layers, n_heads, d_head, d_mlp, dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes)
        )

    def forward(self, images, patch_mask=None):
        """
        images: (B, C, H, W)
        patch_mask: Optional[Tensor] of shape (B, N_patches),
                    where 1 => patch is masked out, 0 => patch is not masked out.
                    If None, no masking is applied.
        """
        x = self.to_patch_embedding(images)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)

        #    patch_mask: shape (B, N), we want shape (B, N+1) after prepending CLS
        if patch_mask is not None:
            # Prepend a zero column for the CLS token (so CLS is not masked)
            patch_mask = F.pad(patch_mask, (1, 0), value=0)  # shape => (B, N+1)
            # We want a 2D attention mask: (N+1) x (N+1) for each batch
            # If row i is masked => entire row = True.
            # If row i is unmasked => only unmasked columns => ~col_mask_bool.
            row_mask = patch_mask.unsqueeze(-1).bool()  # (B, N+1, 1)
            col_mask = patch_mask.unsqueeze(-2).bool()  # (B, 1, N+1)
            attn_mask_2d = (row_mask | (~col_mask)).float()  # (B, N+1, N+1)
            # Finally, expand a "heads" dimension or keep it as (B, 1, N+1, N+1) so it can broadcast:
            attn_mask_2d = attn_mask_2d.unsqueeze(1)  # (B, 1, N+1, N+1)
        else:
            attn_mask_2d = None

        x, attn = self.transformer(x, attn_mask=attn_mask_2d)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        pred_label = self.mlp_head(x)

        return pred_label, attn


if __name__ == '__main__':
    model = ViT(
        image_height=28,
        image_width=28,
        n_channels=1,
        patch_size=7,
        n_classes=10,
        d_model=64,
        n_layers=6,
        n_heads=8,
        d_mlp=256,
        d_head=8,
        pool='cls',
        dropout=0.1,
        emb_dropout=0.1)
