# Model Improvements

## Architecture (`src/models/stgcn.py`)
- **Deeper network**: 3 blocks → 6 blocks `(64, 64, 128↓, 128, 256↓, 256)` — 2x representational depth per resolution level
- **Adaptive graph**: Each block has its own learnable `A_adapt` (`nn.Parameter`, zero-init) added to fixed anatomical A — model learns per-layer graph correction on top of anatomical structure

## Augmentation (`src/dataset.py`)
- **Joint masking** (p=0.3): zero out 10% of nodes — simulates missing MediaPipe detections
- **Temporal shift** (p=0.3): roll ±10 frames with edge-padding
- Existing augmentations kept: horizontal mirror, Gaussian noise, time stretch
- **Fix**: `LandmarkNormalizer` cached as `self._normalizer` in `__init__` instead of recreated per `_augment` call

## Parallel Computing (`src/dataset.py` + `src/train_stgcn.py`)
- `num_workers=4` + `persistent_workers=True` + `prefetch_factor=2` — augmentation parallelized across CPU cores
- AMP (`torch.autocast` + `GradScaler`) — CUDA only, transparent no-op on MPS
- `torch.compile` — CUDA only, skipped on MPS
- `--num-workers` CLI argument added (default: 4)

## Bug Fix (`src/train_stgcn.py`)
- Mixup accuracy was incorrectly measured only against `y_a`; fixed with separate `acc_labels` variable

## Results
| Metric | Baseline (3-block, fixed A) | Improved (6-block, adaptive A) |
|---|---|---|
| Val Top-1 | 29.5% | ~67% |
| Val Top-5 | 60.5% | ~88% |
| Epochs trained | 85 | 100 |
