# Official PyTorch Implementation

- **Venue**: IPCAI 2026
- **Paper**: Optimizing Point-of-Care Ultrasound Video Acquisition for Probabilistic Multi-Task Heart Failure Detection 
- **Authors**: Armin Saadat, Nima Hashemi, Bahar Khodabakhshian, Michael Y. Tsang, Christina Luong, Teresa S.M. Tsang, Purang Abolmaesumi
- **Institution(s)**: University of British Columbia, Vancouver General Hospital


<p align="center">
<a href="https://arxiv.org/abs/2602.13658" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2602.13658-b31b1b.svg?style=flat" /></a>
<a href="https://link.springer.com/article/10.1007/s11548-026-03680-6" alt="springer">
  <img src="https://img.shields.io/badge/Springer-Paper-blue?logo=springer" alt="Springer Paper"> </a>
</p>

## Overview

The paper provides an RL-based framework that optimizes cardiac POCUS video acquisition for heart-failure assessment. The model starts with no observed echo views, sequentially selects the next view to acquire, and can stop once enough information has been collected.

The diagnostic model jointly predicts **aortic stenosis (AS) severity** and **left ventricular ejection fraction (LVEF)** with uncertainty-aware multi-task inference. The acquisition policy is trained to balance diagnostic value against the cost of acquiring more videos.

## Method Summary

- **Inputs**: five standard echo views: AP2, AP3, AP4, PLAX, and PSAX-Ao.
- **Video features**: clips are encoded with a frozen EchoPrime video encoder.
- **Diagnosis**: a task-aware Transformer fuses acquired views and predicts a joint probabilistic output for AS and LVEF.
- **Acquisition policy**: a PPO-based RL agent selects the next view or a stop action using rewards for diagnostic improvement, final correctness, and acquisition cost.

## Paper Highlights

In a retrospective evaluation on **12,180 patient-level studies**, the method matched full-study performance while using fewer videos. At one operating point, it achieved **77.2% mean balanced accuracy (bACC)** across AS and LVEF using about **3.4 videos per study**, corresponding to **32% fewer videos** than full acquisition.

## Method Figure

**Framework overview.** The RL agent selects echo views sequentially, while the frozen encoder and probabilistic diagnostic model update the predicted AS/LVEF distribution after each acquired view.

<img width="1765" height="1079" alt="method_simp" src="https://github.com/user-attachments/assets/9da65194-7c56-4f10-8c46-eea894ddfd8b" />

## Personalized Diagnostic Pathways

**Learned acquisition pathways.** Nodes show partial-view states, patient counts, stopping points, and AS/LVEF bACC for patients diagnosed at each state; edges show the learned view-selection transitions.

<img width="3348" height="1581" alt="DecisionPathways" src="https://github.com/user-attachments/assets/d29d58cd-4d10-4459-a303-33d3b210a191" />


## Installation
To install and run this project locally, please follow these steps:

conda >= 23.11.0

```
git clone repo_address
conda create --name afa python=3.9 pip
conda activate afa
python -m pip install poetry
cd path_to_repo
poetry install
```

## train
```
python run.py --config_path ./configs/default.yaml --save_dir ./logs/run-temp --train
```
Note: In the config file, you can set WANDB_MODE=offline to avoid logging in to WANDB.


## evaluate
```
python run.py --config_path ./configs/default.yaml --save_dir ./logs/eval-run-temp --evaluate
```
Note: In the config file, you can set WANDB_MODE=offline to avoid logging in to WANDB.

## Data and Scope

The paper uses a private echocardiography dataset collected under institutional review board approval, so the dataset is not included in this repository. The method evaluates sequential selection from pre-acquired multi-view echo studies and should be prospectively validated before live bedside POCUS use.

## Citation

If you find this repository useful, please cite the paper:

```bibtex
@inproceedings{saadat2026pocusacquisition,
  title     = {Optimizing Point-of-Care Ultrasound Video Acquisition for Probabilistic Multi-Task Heart Failure Detection},
  author    = {Saadat, Armin and Hashemi, Nima and Khodabakhshian, Bahar and Tsang, Michael Y. and Luong, Christina and Tsang, Teresa S. M. and Abolmaesumi, Purang},
  booktitle = {International Conference on Information Processing in Computer-Assisted Interventions (IPCAI)},
  year      = {2026}
}
```
