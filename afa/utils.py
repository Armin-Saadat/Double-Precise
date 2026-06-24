import os
import pandas as pd
import time
from colorlog import ColoredFormatter
import logging
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, \
    balanced_accuracy_score, mean_absolute_error
import torch
import numpy as np
import random
from einops import rearrange


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For extra determinism (sometimes at a performance cost):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def powerset(lst):
    n = len(lst)
    for i in range(2 ** n):
        subset = [lst[j] for j in range(n) if (i & (1 << j)) > 0]
        if not subset:
            continue
        yield tuple(subset)


def get_classification_metrics(gt: list | np.ndarray, pred: list | np.ndarray):
    bACC = balanced_accuracy_score(gt, pred)
    F1 = f1_score(gt, pred, average='weighted')
    MAE = mean_absolute_error(gt, pred)
    ACC = accuracy_score(gt, pred)
    # recall = recall_score(gt, pred, average=None)
    # precision = precision_score(gt, pred, average=None)
    conf_matrix = confusion_matrix(gt, pred)

    return {'bACC': bACC, 'F1': F1, 'MAE': MAE, 'ACC': ACC, 'Confusion Matrix': conf_matrix}


def plot_confusion_matrix(conf_matrix, dataset_name, save_path=None):
    plt.figure(figsize=(8, 8))    
    if dataset_name in ['AS-EchoInventory']:
        labels = ['no AS', 'early AS', 'significant AS']
    elif dataset_name in ['EF-EchoInventory']:
        labels = ['Reduced', 'Mild', 'Preserved']
    else:
        raise Exception("Unsupported dataset name.")

    y_labels = labels.copy()
    y_labels.reverse()
    reordered_conf_matrix = np.flipud(conf_matrix)

    sns.heatmap(reordered_conf_matrix, annot=True, annot_kws={"size": 28}, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=y_labels)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.xlabel('Predicted Labels', fontsize=17, labelpad=16)
    plt.ylabel('True Labels', fontsize=17, labelpad=16)
    # plt.title('Train Set', fontsize=20, pad=24)
    plt.savefig(save_path) if save_path else plt.show()


def plot_ef_predictions(gt_value: list | np.ndarray, pred_value: list | np.ndarray, gt_category: list | np.ndarray,
                        pred_category: list | np.ndarray, mae=None, bACC=None, save_path=None):
    correct_mask = gt_category == pred_category
    incorrect_mask = ~correct_mask
    plt.figure(figsize=(12, 10))

    # Plot correctly classified samples in blue
    plt.scatter(gt_value[correct_mask], pred_value[correct_mask],
                alpha=0.6, s=50, c='blue', label='Correct Category', edgecolors='navy', linewidth=0.5)

    # Plot misclassified samples in red
    plt.scatter(gt_value[incorrect_mask], pred_value[incorrect_mask],
                alpha=0.8, s=80, c='red', label='Wrong Category', edgecolors='darkred', linewidth=0.8,
                marker='X')

    # Add perfect prediction line (y=x)
    min_val = min(gt_value.min(), pred_value.min())
    max_val = max(gt_value.max(), pred_value.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction', alpha=0.7)

    # Add category boundary lines
    plt.axvline(x=40, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, label='Category Boundaries')
    plt.axvline(x=50, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    plt.axhline(y=40, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    plt.axhline(y=50, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)

    plt.xlabel('Ground Truth EF (%)', fontsize=14, fontweight='bold')
    plt.ylabel('Predicted EF (%)', fontsize=14, fontweight='bold')
    plt.title(
        f'EF Prediction vs Ground Truth\nMAE: {mae:.2f}%, bACC: {100 * bACC:.2f}%',
        fontsize=16, fontweight='bold')
    plt.legend(loc='upper left', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_logger(name: str) -> logging.Logger:
    """
    Creates a custom logger

    :param name: str, name for the logger
    :return: A custom logging.Logger object
    """

    # Define a new level for the logger
    def _info_important(self, msg, *args, **kwargs):
        self.log(logging.INFO + 1, msg, *args, **kwargs)

    formatter = ColoredFormatter(
        "%(log_color)s[%(name)s] %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'white,bold',
            'INFOIMPORTANT': 'cyan,bold',
            'WARNING': 'yellow',
            'ERROR': 'red,bold',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers = [logging.FileHandler(os.path.join(name, 'output.log')), ch]
    logger.propagate = False

    logging.addLevelName(logging.INFO + 1, 'INFOIMPORTANT')
    logging.Logger.info_important = _info_important

    return logger


def get_action_demography(file_path, task):
    episodes_list, episodes_set = {}, {}
    if task == 'AS_EF':
        as_preds, as_labels, ef_preds, ef_labels = [], [], [], []
        with open(file_path, 'r') as file:
            for i, line in enumerate(file):
                if line.split('- ')[0].replace(" ", "") != 'Test':
                    continue
                as_label = int(line.split('- ')[4].replace(" ", "").removeprefix('as_label'))
                as_pred = int(line.split('- ')[5].replace(" ", "").removeprefix('as_pred'))
                as_preds.append(as_pred)
                as_labels.append(as_label)
                ef_label = int(line.split('- ')[6].replace(" ", "").removeprefix('ef_cat_label'))
                ef_pred = int(line.split('- ')[7].replace(" ", "").removeprefix('ef_cat_pred'))
                ef_preds.append(ef_pred)
                ef_labels.append(ef_label)
                episode = tuple(ast.literal_eval(line.split('- ')[3].replace(" ", "").removeprefix('Actions')))

                # order of actions matter
                if episode not in episodes_list:
                    episodes_list[episode] = {'AS1 EF1': 0, 'AS1 EF0': 0, 'AS0 EF1': 0, 'AS0 EF0': 0}
                episodes_list[episode][f'AS{int(as_label == as_pred)} EF{int(ef_label == ef_pred)}'] += 1

                # order of actions does not matter
                episode = tuple(set(episode))
                if episode not in episodes_set:
                    episodes_set[episode] = {'AS1 EF1': 0, 'AS1 EF0': 0, 'AS0 EF1': 0, 'AS0 EF0': 0}
                episodes_set[episode][f'AS{int(as_label == as_pred)} EF{int(ef_label == ef_pred)}'] += 1

    else:
        y_preds, y_trues = [], []
        with open(file_path, 'r') as file:
            for i, line in enumerate(file):
                y_pred = int(line.split('- ')[5].replace(" ", "").removeprefix('y_pred'))
                y_true = int(line.split('- ')[6].replace(" ", "").removeprefix('y_true'))
                result = 'correct' if y_pred == y_true else 'incorrect'
                episode = tuple(ast.literal_eval(line.split('- ')[4].replace(" ", "").removeprefix('Actions')))
                y_preds.append(y_pred)
                y_trues.append(y_true)

                # order of actions matter
                if episode not in episodes_list:
                    episodes_list[episode] = {'correct': 0, 'incorrect': 0}
                episodes_list[episode][result] += 1

                # order of actions does not matter
                episode = tuple(set(episode))
                if episode not in episodes_set:
                    episodes_set[episode] = {'correct': 0, 'incorrect': 0}
                episodes_set[episode][result] += 1

    print('Number of patients:', len(as_preds), end="\n\n")
    print('Considering order of actions')
    for key, value in episodes_list.items():
        print(key, "\t", value)
    print('------------------------------------------', end="\n\n")
    print('Not considering order of actions')
    for key, value in episodes_set.items():
        print(key, "\t", value)

    classification_metrics = get_classification_metrics(as_labels, as_preds)
    for k, v in classification_metrics.items():
        print(f'{k}: {v}')


def mask_patches(images, patch_size, mask_prob=0.5, patch_idx=None):
    """
    images: (B, C, H, W)
    patch_size: int, height/width of each patch
    mask_prob: float, probability of masking each patch
    patch_idx: list, containing the indices of the patches to mask
    returns:
      - patch_mask: (B, N) with 1 => patch is masked out, 0 => not masked
      - masked_images: (B, 1, H, W) grayscale, with masked patches zeroed out

    """

    B, C, H, W = images.shape
    # Number of patches horizontally and vertically
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    N = num_patches_h * num_patches_w  # total patches per image

    # Create a random [B, N] binary mask
    if patch_idx is not None:
        patch_mask = torch.zeros((B, N))
        patch_mask[:, patch_idx] = 1
    else:
        patch_mask = torch.bernoulli(torch.full((B, N), mask_prob))

    # Reshape (patchify) from [B, 1, H, W] -> [B, N, 1, patch_size, patch_size]
    patches = rearrange(
        images,
        'b c (ph p1) (pw p2) -> b (ph pw) c p1 p2',
        p1=patch_size, p2=patch_size
    )

    # Zero out the patches where patch_mask == 1
    #    patch_mask=1 => masked out => patch becomes 0
    #    patch_mask=0 => keep original patch
    mask_expanded = patch_mask.view(B, N, 1, 1, 1)  # shape: (B, N, 1, 1, 1)

    pixel_mask = torch.ones_like(patches)
    patches = patches * (1 - mask_expanded)  # sets to zero where mask=1
    pixel_mask = 1 - (pixel_mask * (1 - mask_expanded))

    # Rearrange patches back to [B, 1, H, W]
    masked_images = rearrange(
        patches,
        'b (ph pw) c p1 p2 -> b c (ph p1) (pw p2)',
        ph=num_patches_h,
        pw=num_patches_w
    )

    pixel_mask = rearrange(
        pixel_mask,
        'b (ph pw) c p1 p2 -> b c (ph p1) (pw p2)',
        ph=num_patches_h,
        pw=num_patches_w
    )

    return patch_mask, pixel_mask, masked_images


def mask_as_study(studies, mask_prob=0.5, video_idx=None):
    """
    studies: (B, N_videos, N_similarities)
    mask_prob: float, probability of masking each video
    video_idx: list, containing the indices of the videos to mask
    returns:
      - mask_idx: (B, N_videos) with 1 => video is masked out, 0 => not masked
      - binary_mask: (B, N_videos, N_similarities), masked values are set to 1, unmasked values set to 0.
      - masked_studies: (B, N_videos, N_similarities), with masked videos zeroed out, others keep the original value
    """

    B, N_vid, N_sim = studies.shape
    device = studies.device

    # Create a random [B, N] binary mask
    if video_idx is not None:
        mask_idx = torch.zeros((B, N_vid)).to(device)
        mask_idx[:, video_idx] = 1
    else:
        mask_idx = torch.bernoulli(torch.full((B, N_vid), mask_prob)).to(device)
    
    # studies with plax or psax count less than 2 have zero embeddings, which would create similarity tensors of 0.5, and should be masked.
    mask_idx = (mask_idx.bool() | (studies == 0.5).all(dim=2).bool()).float()

    # Zero out the videos where mask == 1
    #    mask=1 => masked out => video becomes 0
    #    mask=0 => keep original value
    mask_expanded = mask_idx.view(B, N_vid, 1)  # shape: (B, N, 1)
    masked_studies = studies.clone() * (1 - mask_expanded)

    binary_mask = 1 - (torch.ones_like(studies).to(device) * (1 - mask_expanded))

    return mask_idx, binary_mask, masked_studies


class Profiler:
    def __init__(self, name: str = None, logger: logging.Logger = None):
        self.name = name
        self.logger = logger
        self.times = []
        self.record = False

    def __enter__(self):
        self.record = True
        self.start = time.time()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.record:
            self.times.append(time.time() - self.start)

        return False

    def skip(self):
        self.record = False

    def reset(self):
        self.times = []

    def log(self):
        times_sum = np.sum(self.times)
        times_mean = times_sum / len(self.times) if len(self.times) > 0 else 0
        if self.logger is not None:
            self.logger.warning(
                f'{self.name}: Avg Time {times_mean:.4f} - N_Calls {len(self.times)} - Total Time {times_sum:.1f}')
        else:
            print(f'{self.name}: Avg Time {times_mean:.4f} - N_Calls {len(self.times)} - Total Time {times_sum:.1f}')


def plot_calibration(calibration_metrics):
    """
    Plot reliability diagram with bin colors showing population density

    Args:
        calibration_metrics: Dictionary containing bin information and ECE and MCE
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    bin_data = calibration_metrics['bin_data']

    # Plot the bins
    bin_width = 1.0 / len(bin_data['bin_accuracy'])
    positions = np.linspace(bin_width / 2, 1 - bin_width / 2, len(bin_data['bin_accuracy']))

    # Create color map based on population
    counts = np.array(bin_data['bin_count'])
    max_count = max(counts) if max(counts) > 0 else 1

    # Normalize counts for color mapping
    norm = plt.Normalize(0, max_count)
    cmap = plt.cm.coolwarm  # Blue (low) to Red (high)
    # cmap = plt.cm.RdBu_r  # Reversed Red-Blue (Blue for low, Red for high)

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')

    # Actual calibration with color based on count
    bars = []
    for i, pos in enumerate(positions):
        bar = ax.bar(pos, bin_data['bin_accuracy'][i], width=bin_width,
                     align='center', alpha=0.7, edgecolor='black',
                     color=cmap(norm(counts[i])), label='Accuracy' if i == 0 else "")
        bars.append(bar)

    # Add gap representation
    for i, pos in enumerate(positions):
        if bin_data['bin_count'][i] > 0:  # Only draw gap for non-empty bins
            ax.plot([pos, pos], [bin_data['bin_confidence'][i], bin_data['bin_accuracy'][i]],
                    color='red', linestyle='-', linewidth=2)

    ax.scatter(positions, bin_data['bin_confidence'], color='red', s=50,
               zorder=10, label='Confidence')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title(
        f'Reliability Diagram (ECE={calibration_metrics["ECE"] * 100:.2f}%, MCE={calibration_metrics["MCE"] * 100:.2f}%)')

    # Add a colorbar to show count mapping
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Sample Count')

    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add bin counts
    for i, count in enumerate(bin_data['bin_count']):
        if count > 0:
            ax.text(positions[i], 0.05, f'{count:.1f}', ha='center', fontsize=8)

    plt.tight_layout()


def plot_calibration_to_axis(ax, calibration_metrics, mask_prob):
    """Plot reliability diagram on a given axis"""
    bin_data = calibration_metrics['bin_data']

    # Plot the bins
    bin_width = 1.0 / len(bin_data['bin_accuracy'])
    positions = np.linspace(bin_width / 2, 1 - bin_width / 2, len(bin_data['bin_accuracy']))

    # Create color map based on population
    counts = np.array(bin_data['bin_count'])
    max_count = max(counts) if max(counts) > 0 else 1

    # Normalize counts for color mapping
    norm = plt.Normalize(0, max_count)
    cmap = plt.cm.coolwarm

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')

    # Actual calibration with color based on count
    for i, pos in enumerate(positions):
        bar = ax.bar(pos, bin_data['bin_accuracy'][i], width=bin_width,
                     align='center', alpha=0.7, edgecolor='black',
                     color=cmap(norm(counts[i])))

    # Add gap representation
    for i, pos in enumerate(positions):
        if bin_data['bin_count'][i] > 0:  # Only draw gap for non-empty bins
            ax.plot([pos, pos], [bin_data['bin_confidence'][i], bin_data['bin_accuracy'][i]],
                    color='red', linestyle='-', linewidth=2)

    ax.scatter(positions, bin_data['bin_confidence'], color='red', s=50, zorder=10)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Mask Prob: {mask_prob} (ECE={calibration_metrics["ECE"] * 100:.2f}%)')

    # Add bin counts
    for i, count in enumerate(bin_data['bin_count']):
        if count > 0:
            ax.text(positions[i], 0.05, f'{count:.1f}', ha='center', fontsize=8)

    return norm, cmap, counts  # Return these to create a unified colorbar


def compute_calibration_metrics(y_true, y_pred, confidences, n_bins=10, n_runs=1):
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)

    Args:
        y_true: Ground truth labels
        y_pred: Predicted class labels
        confidences: Predicted confidences (max probability for each prediction)
        n_bins: Number of bins for calibration

    Returns:
        Dictionary containing ECE, MCE, and bin information for plotting
    """
    bin_data = {
        'bin_accuracy': np.zeros(n_bins),
        'bin_confidence': np.zeros(n_bins),
        'bin_count': np.zeros(n_bins),
        'bin_boundaries': np.linspace(0, 1, n_bins + 1)
    }

    # Assign predictions to bins
    for i, conf in enumerate(confidences):
        # Find the bin index for this confidence
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bin_data['bin_count'][bin_idx] += 1
        bin_data['bin_confidence'][bin_idx] += conf
        if y_pred[i] == y_true[i]:
            bin_data['bin_accuracy'][bin_idx] += 1

    # Calculate average accuracy and confidence for each bin
    for i in range(n_bins):
        if bin_data['bin_count'][i] > 0:
            bin_data['bin_accuracy'][i] /= bin_data['bin_count'][i]
            bin_data['bin_confidence'][i] /= bin_data['bin_count'][i]

    # Calculate calibration errors
    bin_weights = bin_data['bin_count'] / np.sum(bin_data['bin_count'])
    bin_errors = np.abs(bin_data['bin_accuracy'] - bin_data['bin_confidence'])

    ece = np.sum(bin_weights * bin_errors)
    mce = np.max(bin_errors)

    bin_data['bin_count'] = bin_data['bin_count'] / n_runs

    return {
        'ECE': ece,
        'MCE': mce,
        'bin_data': bin_data
    }


def calculate_entropy(prob_dist: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Calculate the entropy of a probability distribution.

    Args:
        prob_dist: A tensor containing probability distribution (should sum to 1)
        eps: Small epsilon value to avoid log(0)

    Returns:
        The entropy value as a torch.Tensor
    """
    # Make sure probabilities are valid (avoid numerical issues)
    prob_dist = torch.clamp(prob_dist, min=eps)

    # Normalize if needed
    if torch.abs(prob_dist.sum() - 1.0) > eps:
        prob_dist = prob_dist / prob_dist.sum()

    # Calculate entropy: H(p) = -∑ p_i * log(p_i)
    entropy = -torch.sum(prob_dist * torch.log(prob_dist))

    return entropy


def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Calculate the Kullback-Leibler divergence between distributions p and q.

    Args:
        p: First probability distribution (tensor)
        q: Second probability distribution (tensor)
        eps: Small epsilon value to avoid log(0)

    Returns:
        KL divergence as a torch.Tensor
    """
    # Clamp values to avoid numerical issues
    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)

    # Normalize if needed
    if torch.abs(p.sum() - 1.0) > eps:
        p = p / p.sum()
    if torch.abs(q.sum() - 1.0) > eps:
        q = q / q.sum()

    # KL divergence: KL(p||q) = Σ p_i * log(p_i/q_i)
    return torch.sum(p * torch.log(p / q))


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Calculate the Jensen-Shannon divergence between distributions p and q.

    Args:
        p: First probability distribution (tensor)
        q: Second probability distribution (tensor)
        eps: Small epsilon value to avoid numerical issues

    Returns:
        JS divergence as a torch.Tensor (value between 0 and 1)
    """
    # Ensure tensors are on the same device
    if p.device != q.device:
        q = q.to(p.device)

    # Calculate the average distribution m
    m = 0.5 * (p + q)

    # JS divergence is the average of KL(p||m) and KL(q||m)
    js_div = 0.5 * kl_divergence(p, m, eps) + 0.5 * kl_divergence(q, m, eps)

    return js_div


def js_distance(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Calculate the Jensen-Shannon distance (square root of JS divergence).
    This is a proper metric that satisfies the triangle inequality.

    Args:
        p: First probability distribution (tensor)
        q: Second probability distribution (tensor)
        eps: Small epsilon value to avoid numerical issues

    Returns:
        JS distance as a torch.Tensor (value between 0 and 1)
    """
    return torch.sqrt(js_divergence(p, q, eps))


def ef_category(ef_values):
    """Convert EF regression values to categories for bACC calculation"""
    categories = torch.zeros_like(ef_values, dtype=torch.long)
    categories[ef_values < 40] = 0  # Class 0: EF < 40%
    categories[(ef_values >= 40) & (ef_values <= 50)] = 1  # Class 1: 40% ≤ EF ≤ 50%
    categories[ef_values > 50] = 2  # Class 2: EF > 50%
    return categories
