# Implementation Details

This document provides the full implementation and optimization settings for
UMGRL. The paper reports only the settings needed to interpret the results;
everything required to reproduce them is listed here.

## Hardware and framework

| Item | Setting |
|---|---|
| GPU | NVIDIA RTX A6000 (48 GB) |
| Framework | PyTorch, built on OpenGait |

## Stage 1: Modality-robust pretraining

Stage 1 is pretrained on SUSTech1K and CCPG with varying modality availability.

### Key settings

| Item | Setting |
|---|---|
| Pretraining datasets | SUSTech1K, CCPG |
| Part-wise tokens / queries | P = 16 |
| Loss weights (λ1, λ2, λ3) | (1, 1, 1) |
| Modality dropout: keep all modalities | 0.2 |
| Modality dropout: drop one modality | 0.7 |
| Modality dropout: drop two modalities | 0.1 |
| Modality dropout for two-modality samples (CCPG): keep both / drop one | 0.3 / 0.7 |
| LidarGait++ part tokens | 31, linearly resampled to 16 |

### Optimization

| Item | Setting |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Weight decay | 1e-5 |
| Batch size | 64 |
| Iterations | 50k |
| LR scheduler | MultiStepLR, milestones at 20k / 30k / 40k, decay factor 0.1 |
| Precision | Mixed precision (AMP) |

### Cross-modal imputer

| Item | Setting |
|---|---|
| Architecture | Transformer decoder |
| Decoder layers | 2 |
| Hidden dimension | 256 |
| Attention heads | 4 |
| FFN dimension | 1024 |
| Activation | GELU |
| Dropout | 0.1 |
| Normalization | Post-LN |
| Part-wise tokens | 16 |
| Learnable queries per target modality | 16 |
| Parameter sharing | One decoder shared across all target modalities |

## Stage 2: Task-specific learning

All pretrained Stage 1 representation-learning modules are frozen, except the
final linear projection layer of the Shared Latent Projector.

Scoliosis1K provides silhouettes and 2D pose but no paired LiDAR point clouds.
The main downstream experiments therefore use silhouette and 2D pose as the
observed inputs. Point-cloud information is used during Stage 1 pretraining and
is evaluated downstream through the predicted task-ready representations.

### Key settings

| Item | Setting |
|---|---|
| Downstream dataset | Scoliosis1K |
| Observed modalities | Silhouette, 2D pose |
| Trainable Stage 1 parameters | Final linear projection of the Shared Latent Projector |
| K | 4 |
| λ4 | 1 |
| λ5 | 0.1 |
| λ_KL | 0.01 |

### Optimization

| Item | Setting |
|---|---|
| Optimizer | Adam |
| Learning rate | 3e-4 |
| Weight decay | 1e-5 |
| Iterations | 40k |
| LR scheduler | MultiStepLR, milestones at 10k / 20k / 30k |
| KL annealing delay | 3,000 iterations |
| KL annealing length | 20,000 iterations |
