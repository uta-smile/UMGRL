# UMGRL

Official repository for **Uncertainty-Aware Multimodal Gait Representation Learning for Scoliosis Screening** (UMGRL), accepted to *IEEE Transactions on Medical Imaging*.

UMGRL is the journal extension of our MICCAI 2025 paper, TG-MILNet (*Text-Guided Multi-Instance Learning for Scoliosis Screening via Gait Video Analysis*).

This repository contains:

- **Stage 1: modality-robust pretraining.** Silhouette, 2D pose, and LiDAR point-cloud features are projected into a shared latent space, decomposed into modality-invariant and modality-specific components, aligned on the invariant component, and recomposed through a learnable gate. A teacher–student branch with modality dropout learns to predict the task-ready representation of a missing modality from the available ones.
- **DRF reproduction.** Our reimplementation of DRF (MICCAI 2025), used as a baseline on Scoliosis1K.

## Repository structure

```
configs/
  stage1/          encoder configs used to extract Stage 1 input features
  drf/             DRF configs for the 1:1:2, 1:1:4, 1:1:8, and 1:1:16 splits
opengait/          training framework (based on OpenGait)
  modeling/models/unified_encoder.py   Stage 1 model
  modeling/losses/geo_alignment.py     Stage 1 losses
  modeling/models/drf.py               DRF model
train_stage1.py    Stage 1 training
docs/              full implementation details
```

## Requirements

Tested with Python 3.9, PyTorch 2.4, and CUDA 12.1.

```bash
pip install -r requirements.txt
```

## Data preparation

Obtain SUSTech1K, CCPG, and Scoliosis1K from their official sources and convert them to the OpenGait pickle format: silhouettes, 2D pose heatmaps, and, for SUSTech1K, LiDAR point clouds.

In each config under `configs/`, set `dataset_root` to the converted data. `dataset_partition` points to the train/test partition file of each dataset:

| Dataset | Partition file |
|---|---|
| SUSTech1K | `datasets/SUSTech1K/SUSTech1K.json` |
| CCPG | `datasets/CCPG/CCPG.json` |
| Scoliosis1K | `datasets/Scoliosis1K/Scoliosis1K_<ratio>.json`, with `<ratio>` in `112`, `114`, `118`, `1116` |

All commands below are run from the repository root.

## Stage 1: modality-robust pretraining

Stage 1 is trained on features produced by three pretrained gait encoders: GaitBase for silhouettes, DeepGaitV2 for 2D pose heatmaps, and LidarGait++ for point clouds. SUSTech1K provides all three modalities; CCPG provides silhouette and 2D pose.

**1. Train the encoders.**

```bash
for cfg in gaitbase_sustech1k deepgaitv2_sustech1k lidargaitv2_sustech1k gaitbase_ccpg deepgaitv2_ccpg; do
  python -m torch.distributed.run --nproc_per_node=1 opengait/main.py \
      --cfgs configs/stage1/${cfg}.yaml --phase train
done
```

**2. Extract features.** Running the test phase saves one `(1, 256, P)` feature per sequence under `output/<dataset>/<model>/<save_name>/embeddings`.

```bash
for cfg in gaitbase_sustech1k deepgaitv2_sustech1k lidargaitv2_sustech1k gaitbase_ccpg deepgaitv2_ccpg; do
  python -m torch.distributed.run --nproc_per_node=1 opengait/main.py \
      --cfgs configs/stage1/${cfg}.yaml --phase test
done
```

**3. Train Stage 1.**

```bash
python train_stage1.py --embedding_root output --save_dir output/UnifiedEncoder
```

Each iteration runs one forward pass over all modalities available for a sample. The representation loss is computed on this pass:

L_repr = L_geo + λ1 · L_dir + λ2 · L_ortho

- `L_geo` encourages the pooled invariant descriptors of all modalities to form a rank-1 matrix.
- `L_dir` enforces pairwise cosine agreement between them.
- `L_ortho` separates the invariant and modality-specific components.

The detached task-ready representations of this pass are the teacher targets. Modalities are then dropped per sample (keep all / drop one / drop two = 0.2 / 0.7 / 0.1), and the imputer predicts the dropped representations from the visible ones:

L_stage1 = L_repr + λ3 · L_imp

Defaults: Adam, learning rate 1e-4, weight decay 1e-5, batch size 64, 50k iterations, learning rate decayed by 0.1 at 20k, 30k, and 40k, and (λ1, λ2, λ3) = (1, 1, 1). Checkpoints are written to `--save_dir` every 10k iterations.

The trained model exposes `get_final_representation(z_seg, z_pose, z_cloud)`, which returns the task-ready representation of every observed modality and imputes the missing ones.

## DRF reproduction

DRF takes two inputs derived from 2D pose: a skeleton heatmap and PAV, a per-sequence descriptor of left–right keypoint asymmetry. Each Scoliosis1K sequence directory under `dataset_root` holds `0_heatmap.pkl` with the heatmaps and `1_pav.pkl` with the PAV repeated for every frame.

Train and evaluate on each class-imbalance split (`112`, `114`, `118`, `1116`):

```bash
python -m torch.distributed.run --nproc_per_node=1 opengait/main.py --cfgs configs/drf/drf_112.yaml --phase train
python -m torch.distributed.run --nproc_per_node=1 opengait/main.py --cfgs configs/drf/drf_112.yaml --phase test
```

The test phase reports accuracy, macro-F1, macro-AUC, and per-class recall, precision, and specificity for the checkpoint set by `evaluator_cfg.restore_hint`.

## Implementation details

Full implementation and optimization settings are listed in [docs/implementation_details.md](docs/implementation_details.md).
## Acknowledgements and Citations

This repository builds upon the [OpenGait](https://github.com/ShiqiYu/OpenGait) framework and uses the Scoliosis1K dataset for downstream evaluation. We sincerely thank the OpenGait authors and contributors for providing the gait-recognition platform, and the Scoliosis1K authors for releasing the dataset and associated resources.

If you use this repository, please consider citing the following related works:

```bibtex
@inproceedings{fan2023opengait,
  title={OpenGait: Revisiting Gait Recognition Towards Better Practicality},
  author={Fan, Chao and Liang, Junhao and Shen, Chuanfu and Hou, Saihui and Huang, Yongzhen and Yu, Shiqi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9707--9716},
  year={2023}
}

@inproceedings{zhou2024gait,
  title={Gait Patterns as Biomarkers: A Video-Based Approach for Classifying Scoliosis},
  author={Zhou, Zirui and Liang, Junhao and Peng, Zizhao and Fan, Chao and An, Fengwei and Yu, Shiqi},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={284--294},
  year={2024},
  organization={Springer}
}

@inproceedings{zhou2025gait,
  title={Pose as Clinical Prior: Learning Dual Representations for Scoliosis Screening},
  author={Zhou, Zirui and Peng, Zizhao and Jin, Dongyang and Fan, Chao and An, Fengwei and Yu, Shiqi},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={467--476},
  year={2025},
  organization={Springer}
}
