import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from einops import repeat

from afa.classifiers.mnist_vit import Transformer
from afa.utils import mask_as_study

ACTION_2_VIDEO_IDX = {0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]}


class ConfidentMultiFormer(nn.Module):
    def __init__(self, *, n_videos=5, n_sims=512, d_model=32, n_layers=3, n_heads=4,
                 d_mlp=128, d_head=4, pool='cls', dropout=0.3, emb_dropout=0.1, prob_func='PDF', dist='gaussian',
                 sigma_floor=0.05, cov_type='diag', mu_activation='sigmoid', logger=None):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.n_videos = n_videos
        self.pool = pool
        self.n_tasks = 2  # AS and EF
        self.cov_type = cov_type
        self.sigma_floor = sigma_floor

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
        self.pos_embedding = nn.Parameter(torch.randn(1, n_videos + self.n_tasks, d_model))
        self.dropout = nn.Dropout(emb_dropout)

        # Shared transformer backbone
        self.transformer = Transformer(d_model, n_layers, n_heads, d_head, d_mlp, dropout)

        if mu_activation == 'sigmoid':
            self.mu = nn.Sequential(
                nn.LayerNorm(d_model * self.n_tasks),
                nn.Linear(d_model * self.n_tasks, self.n_tasks),
                nn.Sigmoid()
            )
        elif mu_activation is None:
            self.mu = nn.Sequential(
                nn.LayerNorm(d_model * self.n_tasks),
                nn.Linear(d_model * self.n_tasks, self.n_tasks),
            )
        else:
            raise Exception(f"Unsupported mu_activation: {mu_activation}")

        if self.cov_type == 'diag':
            self.sigma = nn.Sequential(
                nn.LayerNorm(d_model * self.n_tasks),
                nn.Linear(d_model * self.n_tasks, self.n_tasks),
                nn.Softplus()
            )
        elif self.cov_type == 'full':
            n_cov_params = self.n_tasks * (self.n_tasks + 1) // 2
            self.sigma = nn.Sequential(
                nn.LayerNorm(d_model * self.n_tasks),
                nn.Linear(d_model * self.n_tasks, n_cov_params),
            )
        else:
            raise Exception(f"Unsupported cov_type: {cov_type}")

        # AS bounds
        as_bounds_values = torch.tensor([0.0, 0.333333, 0.666666, 1.0])  # 0.0 normal 0.33 early 0.66 severe 1.0
        as_bounds = torch.stack((as_bounds_values[:-1], as_bounds_values[1:]), dim=1)

        # EF bounds
        ef_bounds_values = torch.tensor([0.2, 0.4, 0.5, 0.7])  # 0.1 reduced 0.4 mild 0.5 preserved 0.8
        ef_bounds = torch.stack((ef_bounds_values[:-1], ef_bounds_values[1:]), dim=1)

        if prob_func == 'CDF':
            pass
        elif prob_func == 'PDF':
            self.prob_func = MultiTaskPDFProb(bounds_list=[as_bounds, ef_bounds], cov_type=self.cov_type)
        else:
            raise Exception(f"Unsupported prob_func: {prob_func}")

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
            mask_idx = F.pad(mask_idx, (self.n_tasks, 0), value=0)  # shape => (B, N+2)

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
            pooled_as = x.mean(dim=1)
            pooled_ef = x.mean(dim=1)
        else:
            pooled_as = x[:, 0]  # First CLS token for AS
            pooled_ef = x[:, 1]  # Second CLS token for EF

        joint_cls = torch.cat((pooled_as, pooled_ef), dim=1)

        y_mu = self.mu(joint_cls)  # (B, n_tasks)

        if self.cov_type == 'diag':
            y_sigma = self.sigma(joint_cls) + self.sigma_floor  # (B, n_tasks)
        elif self.cov_type == 'full':
            # Build full covariance matrix from lower triangular
            y_sigma = self._build_covariance_matrix(self.sigma(joint_cls))  # (B, n_tasks, n_tasks)
        else:
            raise Exception(f"Unsupported cov_type: {self.cov_type}")

        # joint: (B, n_as_classes * n_ef_classes) and marginal: [(B, n_as_classes), (B, n_ef_classes)]
        joint_probs, marginal_probs = self.prob_func(y_mu, y_sigma)

        out = {'joint_probs': joint_probs,
               'marginal_probs': marginal_probs,
               'mu': y_mu,
               'sigma': y_sigma,
               'attn': attn}

        return out

    def _build_covariance_matrix(self, lower_tri_params):
        """
        Build a full covariance matrix from lower triangular parameters.
        
        Args:
            lower_tri_params: (B, n_params) where n_params = n_tasks * (n_tasks + 1) // 2
            
        Returns:
            cov_matrix: (B, n_tasks, n_tasks) positive semi-definite covariance matrix
        """
        bs = lower_tri_params.shape[0]
        
        # Create lower triangular matrix L
        L = torch.zeros(bs, self.n_tasks, self.n_tasks, device=lower_tri_params.device)
        
        # Fill in the lower triangular part
        tril_indices = torch.tril_indices(self.n_tasks, self.n_tasks, offset=0)
        L[:, tril_indices[0], tril_indices[1]] = lower_tri_params
        
        # Apply softplus to diagonal elements to ensure they're positive (avoiding in-place)
        diag_mask = torch.eye(self.n_tasks, device=L.device).bool()
        diag_elements = L[:, diag_mask]  # Extract diagonal (B, n_tasks)
        diag_elements = F.softplus(diag_elements) + self.sigma_floor
        
        # Reconstruct L with processed diagonal
        L_new = L.clone()
        L_new[:, diag_mask] = diag_elements
        
        # Compute covariance matrix as L @ L.T (ensures positive semi-definite)
        cov_matrix = torch.bmm(L_new, L_new.transpose(1, 2))
        
        return cov_matrix

    def predict(self, state: torch.tensor, action_history: list):
        """
        state: torch.tensor, shape: [N_videos, N_similarities] or [N_videos, D_embedding]
        action_history: list
        """
        assert 0 not in action_history, f"0 is a termination action, so the process should have been terminated."

        observed_videos_idx = set([idx for a in action_history for idx in ACTION_2_VIDEO_IDX[a]])
        masked_videos_idx = list(set(range(self.n_videos)) - observed_videos_idx)

        with torch.no_grad():
            mask_idx, _, masked_studies = mask_as_study(state.unsqueeze(0), mask_prob=0.0, video_idx=masked_videos_idx)
            out = self.forward(studies=masked_studies, mask_idx=mask_idx)
            as_pred_probs = out['marginal_probs'][0]
            as_pred = torch.argmax(as_pred_probs, dim=1)
            ef_mu = out['mu'][:, 1] * 100
            ef_pred_probs = out['marginal_probs'][1]
            ef_pred_category = torch.argmax(ef_pred_probs, dim=1)

        output = {
            'class_pred': as_pred.squeeze(0).cpu().numpy(),
            'reg_pred': ef_mu.squeeze(0).cpu().numpy(),
            'reg_pred_category': ef_pred_category.squeeze(0).cpu().numpy(),
            'joint_probs': out['joint_probs'].squeeze(0).cpu(),
            'class_pred_probs': as_pred_probs.squeeze(0).cpu().numpy(),
            'reg_pred_probs': ef_pred_probs.squeeze(0).cpu().numpy(),
        }

        return output


class MultiTaskPDFProb(nn.Module):
    """
    Converts (ŷ, Σ̂) -> joint class probs for multiple tasks using multivariate Gaussian.

    For tasks with n_classes = [n1, n2, ..., nk], this creates all possible class combinations
    and evaluates the multivariate Gaussian PDF at each combination center.

    Args:
        bounds_list: List of bounds tensors, one per task
                    Each bounds[i] has shape (n_classes_i, 2) with [lower, upper] bounds
        cov_type: 'diag' for diagonal covariance or 'full' for full covariance matrix
    """

    def __init__(self, bounds_list, cov_type='diag'):
        super().__init__()
        self.n_tasks = len(bounds_list)
        self.cov_type = cov_type

        # Compute class centers for each task
        centres_list = [bounds.mean(dim=1) for bounds in bounds_list]
        self.n_classes_per_task = [len(c) for c in centres_list]

        # Create all combinations of class centers (Cartesian product)
        # Result shape: (total_combinations, n_tasks)
        meshgrid = torch.meshgrid(*centres_list, indexing='ij')
        joint_centres = torch.stack([grid.flatten() for grid in meshgrid], dim=1)

        self.register_buffer("joint_centres", joint_centres)  # (n_combinations, n_tasks)
        self.register_buffer("n_classes_per_task_tensor", torch.tensor(self.n_classes_per_task))

    def forward(self, y_mu, y_sigma):
        """
        Args:
            y_mu: Mean predictions, shape (B, n_tasks)
            y_sigma: Standard deviation predictions, shape (B, n_tasks) for 'diag'
                     or (B, n_tasks, n_tasks) for 'full'

        Returns:
            joint_probs: Joint probabilities, shape (B, n_combinations)
            marginal_probs: List of marginal probabilities for each task
                           marginal_probs[i] has shape (B, n_classes_i)
        """
        bs = y_mu.shape[0]
        y_mu = y_mu.clamp(min=0.0, max=1.0)
        y_sigma = y_sigma.clamp_min(1e-3)

        if self.cov_type == 'diag':
            # Diagonal covariance: independent tasks
            cov_matrix = torch.diag_embed(y_sigma ** 2)  # (B, n_tasks, n_tasks)
        else:
            # Full covariance matrix (assumes y_sigma is already (B, n_tasks, n_tasks))
            cov_matrix = y_sigma

        # Evaluate multivariate Gaussian at each joint center
        log_probs = []
        for i in range(self.joint_centres.shape[0]):
            center = self.joint_centres[i:i + 1]  # (1, n_tasks)
            dist = MultivariateNormal(y_mu, cov_matrix)
            log_prob = dist.log_prob(center.expand(bs, -1))  # (B,)
            log_probs.append(log_prob)

        log_probs = torch.stack(log_probs, dim=1)  # (B, n_combinations)

        # Convert log probs to probs and normalize
        joint_probs = torch.exp(log_probs)
        joint_probs = joint_probs / joint_probs.sum(dim=1, keepdim=True)

        # Compute marginal probabilities for each task
        marginal_probs = self._compute_marginals(joint_probs)

        return joint_probs, marginal_probs

    def _compute_marginals(self, joint_probs):
        """
        Marginalize joint probabilities to get per-task probabilities.

        Args:
            joint_probs: (B, n_combinations)

        Returns:
            List of marginal probabilities, one per task
        """
        bs = joint_probs.shape[0]

        # Reshape joint probs to (B, n_classes_0, n_classes_1, ..., n_classes_k)
        shape = [bs] + self.n_classes_per_task
        joint_probs_reshaped = joint_probs.view(*shape)

        marginals = []
        for task_idx in range(self.n_tasks):
            # Sum over all dimensions except batch and current task
            dims_to_sum = list(range(1, self.n_tasks + 1))
            dims_to_sum.remove(task_idx + 1)
            marginal = joint_probs_reshaped.sum(dim=dims_to_sum)
            marginals.append(marginal)

        return marginals

if __name__ == '__main__':
    model = ConfidentMultiFormer(
        n_videos=5,
        n_sims=512,
        d_model=64,
        n_layers=6,
        n_heads=8,
        d_mlp=256,
        d_head=8,
        pool='cls',
        dropout=0.1,
        emb_dropout=0.1,
        prob_func='PDF',
        dist='gaussian',
        sigma_floor=0.05,
        mu_activation='sigmoid',
        cov_type='diag')

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_nontrainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print("Trainable parameters:", total_trainable_params)
    print("Non-trainable parameters:", total_nontrainable_params)

    bs = 10
    study = torch.rand(bs, 5, 512)
    out = model(study)

    print(f"\n{'Forward Pass Results':^80}")
    print("-" * 80)
    print(f"Input shape: {study.shape}")
    print(f"Joint probability shape: {out['joint_probs'].shape}")
    print(f"Number of marginal distributions: {len(out['marginal_probs'])}")
    print(f"Task A marginal shape: {out['marginal_probs'][0].shape}")
    print(f"Task B marginal shape: {out['marginal_probs'][1].shape}")

    print(f"\n{'Sample Predictions':^80}")
    print("-" * 80)
    for i in range(bs):
        print(f"\nSample {i + 1}:")
        print(f"  Joint probabilities: {out['joint_probs'][i].detach().numpy()}")
        print(f"  Task A marginal: {out['marginal_probs'][0][i].detach().numpy()}")
        print(f"  Task B marginal: {out['marginal_probs'][1][i].detach().numpy()}")
        print(f"  Task A prediction: Class {torch.argmax(out['marginal_probs'][0][i]).item()}")
        print(f"  Task B prediction: Class {torch.argmax(out['marginal_probs'][1][i]).item()}")

    # print("\n" + "=" * 80)
    # print("JOINT PROBABILITY INTERPRETATION GUIDE")
    # print("=" * 80)
    # print("If Task A has 3 classes and Task B has 2 classes:")
    # print("The 6 joint probabilities correspond to all combinations:")
    # print("  [0] = P(Task A=class 0, Task B=class 0)")
    # print("  [1] = P(Task A=class 0, Task B=class 1)")
    # print("  [2] = P(Task A=class 1, Task B=class 0)")
    # print("  [3] = P(Task A=class 1, Task B=class 1)")
    # print("  [4] = P(Task A=class 2, Task B=class 0)")
    # print("  [5] = P(Task A=class 2, Task B=class 1)")
    # print("\nMarginal probabilities are computed by summing over the other task's classes.")
