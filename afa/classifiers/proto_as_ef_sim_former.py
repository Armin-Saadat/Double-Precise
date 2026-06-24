import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from afa.classifiers.mnist_vit import Transformer
from afa.utils import mask_as_study

ACTION_2_VIDEO_IDX = {0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]}


class ProtoASEFSimFormer(nn.Module):
    def __init__(self, *, n_videos=5, n_sims=512, as_n_classes=3, ef_n_classes=1, d_model=32, n_layers=3, n_heads=4,
                 d_mlp=128, d_head=4, pool='cls', dropout=0.3, emb_dropout=0.1, logger=None):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.n_videos = n_videos
        self.pool = pool

        # Shared embedding layer for video features
        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(n_sims),
            nn.Linear(n_sims, d_model),
            nn.LayerNorm(d_model),
        )

        # Two CLS tokens: one for AS, one for EF
        self.cls_token_as = nn.Parameter(torch.randn(1, 1, d_model))
        self.cls_token_ef = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional embeddings for 2 CLS tokens + n_videos
        self.pos_embedding = nn.Parameter(torch.randn(1, n_videos + 2, d_model))
        self.dropout = nn.Dropout(emb_dropout)

        # Shared transformer backbone
        self.transformer = Transformer(d_model, n_layers, n_heads, d_head, d_mlp, dropout)

        # Task-specific MLP heads
        # AS classification head (3 classes)
        self.as_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, as_n_classes)
        )

        # EF regression/classification head
        self.ef_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ef_n_classes),
        )

    def forward(self, studies, mask_idx=None):
        """
        studies: (B, N_videos, N_similarities)
        mask_idx: Optional[Tensor] of shape (B, N_videos),
                    where 1 => video is masked out, 0 => video is not masked out.
                    If None, no masking is applied.
        
        Returns:
            as_logits: (B, as_n_classes) - AS classification logits
            ef_predictions: (B, 1) - EF regression predictions (0-100 scale)
            attn: attention weights from transformer
        """
        x = self.to_patch_embedding(studies)
        b, n, _ = x.shape

        # Concatenate both CLS tokens
        cls_tokens_as = repeat(self.cls_token_as, '1 1 d -> b 1 d', b=b)
        cls_tokens_ef = repeat(self.cls_token_ef, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens_as, cls_tokens_ef, x), dim=1)  # (B, 2+N_videos, d_model)

        x += self.pos_embedding
        x = self.dropout(x)

        # Handle masking - prepend zeros for both CLS tokens
        if mask_idx is not None:
            # Prepend two zero columns for the CLS tokens (so CLS tokens are not masked)
            mask_idx = F.pad(mask_idx, (2, 0), value=0)  # shape => (B, N+2)

            # Create 2D attention mask
            row_mask = mask_idx.unsqueeze(-1).bool()  # (B, N+2, 1)
            col_mask = mask_idx.unsqueeze(-2).bool()  # (B, 1, N+2)
            attn_mask_2d = (row_mask | (~col_mask)).float()  # (B, N+2, N+2)
            attn_mask_2d = attn_mask_2d.unsqueeze(1)  # (B, 1, N+2, N+2)
        else:
            attn_mask_2d = None

        # Pass through transformer
        x, attn = self.transformer(x, attn_mask=attn_mask_2d)

        if self.pool == 'mean':
            # Mean pooling over all tokens
            pooled_as = x.mean(dim=1)
            pooled_ef = x.mean(dim=1)
        else:
            # Use respective CLS tokens
            pooled_as = x[:, 0]  # First CLS token for AS
            pooled_ef = x[:, 1]  # Second CLS token for EF

        # Task-specific predictions
        as_logits = self.as_head(pooled_as)
        ef_logits = self.ef_head(pooled_ef)

        return as_logits, ef_logits, attn

    def predict(self, state: torch.tensor, action_history: list):
        """
        state: torch.tensor, shape: [N_videos, N_similarities] or [N_videos, D_embedding]
        action_history: list
        
        Returns:
            as_pred: AS prediction class
            ef_pred: EF prediction value
            as_logits: AS logits
            ef_logits: EF logits
            attn: attention weights
        """
        assert 0 not in action_history, f"0 is a termination action, so the process should have been terminated."

        observed_videos_idx = set([idx for a in action_history for idx in ACTION_2_VIDEO_IDX[a]])
        masked_videos_idx = list(set(range(self.n_videos)) - observed_videos_idx)

        with torch.no_grad():
            mask_idx, _, masked_studies = mask_as_study(state.unsqueeze(0), mask_prob=0.0, video_idx=masked_videos_idx)
            as_logits, ef_logits, attn = self.forward(studies=masked_studies, mask_idx=mask_idx)
            _, as_pred = torch.max(as_logits, dim=1)

        output = {'class_pred': as_pred.squeeze(0).cpu().numpy(),
                  'reg_pred': ef_logits.squeeze(0).cpu().numpy(),
                  }

        return output


if __name__ == '__main__':
    model = ProtoASEFSimFormer(
        n_videos=5,
        n_sims=512,
        as_n_classes=3,
        d_model=64,
        n_layers=6,
        n_heads=8,
        d_mlp=256,
        d_head=8,
        pool='cls',
        dropout=0.1,
        emb_dropout=0.1)

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_nontrainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print("Trainable parameters:", total_trainable_params)
    print("Non-trainable parameters:", total_nontrainable_params)

    # Test forward pass
    batch_size = 2
    n_videos = 5
    n_sims = 512

    dummy_studies = torch.randn(batch_size, n_videos, n_sims)
    as_logits, ef_predictions, attn = model(dummy_studies)

    print(f"AS logits shape: {as_logits.shape}")  # Should be (2, 3)
    print(f"EF predictions shape: {ef_predictions.shape}")  # Should be (2, 1)
    print(f"EF predictions range: {ef_predictions.min().item():.2f} - {ef_predictions.max().item():.2f}")
