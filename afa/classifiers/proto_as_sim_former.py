import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from afa.classifiers.mnist_vit import Transformer
from afa.utils import mask_as_study

ACTION_2_VIDEO_IDX = {0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]}


class ProtoASSimFormer(nn.Module):
    def __init__(self, *, n_videos=4, n_sims=40, n_classes=3, d_model=32, n_layers=3, n_heads=4, d_mlp=128, d_head=4,
                 pool='cls', dropout=0.3, emb_dropout=0.1, logger=None):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.n_videos = n_videos

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(n_sims),
            nn.Linear(n_sims, d_model),
            nn.LayerNorm(d_model),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, n_videos + 1, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(d_model, n_layers, n_heads, d_head, d_mlp, dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes)
        )

    def forward(self, studies, mask_idx=None):
        """
        studies: (B, N_videos, N_similarities)
        mask_idx: Optional[Tensor] of shape (B, N_videos),
                    where 1 => video is masked out, 0 => video is not masked out.
                    If None, no masking is applied.
        """
        x = self.to_patch_embedding(studies)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)

        #    patch_mask: shape (B, N), we want shape (B, N+1) after prepending CLS
        if mask_idx is not None:
            # Prepend a zero column for the CLS token (so CLS is not masked)
            mask_idx = F.pad(mask_idx, (1, 0), value=0)  # shape => (B, N+1)
            # We want a 2D attention mask: (N+1) x (N+1) for each batch
            # If row i is masked => entire row = True.
            # If row i is unmasked => only unmasked columns => ~col_mask_bool.
            row_mask = mask_idx.unsqueeze(-1).bool()  # (B, N+1, 1)
            col_mask = mask_idx.unsqueeze(-2).bool()  # (B, 1, N+1)
            attn_mask_2d = (row_mask | (~col_mask)).float()  # (B, N+1, N+1)
            # Finally, expand a "heads" dimension or keep it as (B, 1, N+1, N+1) so it can broadcast:
            attn_mask_2d = attn_mask_2d.unsqueeze(1)  # (B, 1, N+1, N+1)
        else:
            attn_mask_2d = None

        x, attn = self.transformer(x, attn_mask=attn_mask_2d)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        pred_logits = self.mlp_head(x)

        return pred_logits, attn

    def predict(self, state: torch.tensor, action_history: list):
        """
            state: torch.tensor, shape: [N_videos, N_similarities]
            action_history: list
        """
        assert 0 not in action_history, f"0 is a termination action, so the process should have been terminated."

        observed_videos_idx = set([idx for a in action_history for idx in ACTION_2_VIDEO_IDX[a]])
        masked_videos_idx = list(set(range(self.n_videos)) - observed_videos_idx)

        with torch.no_grad():
            mask_idx, _, masked_studies = mask_as_study(state.unsqueeze(0), mask_prob=0.0, video_idx=masked_videos_idx)
            pred_logits, attn = self.forward(studies=masked_studies, mask_idx=mask_idx)
            _, y_pred = torch.max(pred_logits, dim=1)

        output = {'class_pred': y_pred.squeeze(0).cpu().numpy(),
                  'reg_pred': pred_logits.squeeze(0).cpu().numpy(),
                  }

        return output


if __name__ == '__main__':
    model = ProtoASSimFormer(
        n_videos=4,
        n_sims=40,
        n_classes=3,
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

    bs = 16
    n_videos = 4
    video_emb = 40  # same as n_sims in this model
    study = torch.randn([bs, n_videos, video_emb])

    mask_idx, binary_mask, masked_study = mask_as_study(study, mask_prob=0.0, video_idx=None)
    pred_probs, attn, *_ = model(masked_study, mask_idx=mask_idx)
