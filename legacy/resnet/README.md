# Legacy: ResNet-18 fine-tuning

This was the first version of the project, before the switch to a frozen
DINOv2 backbone with linear probes (see the main [README](../../README.md)).
Each task fine-tuned its own ImageNet-pretrained ResNet-18
(`scripts/train_*.py`, with quick test scripts in `tests/`).

It is kept for reference only:

- **Weights are not included** (≈250 MB of `.pth` / `.onnx` files).
- **The scripts do not run as-is.** They import `set_bn_eval` and
  `grouped_split` from an older `scripts/common.py` that no longer exists.
- `results/` holds the last retraining logs and `graphics/` the training curves.

## Why it was replaced

- **Misleading evaluation.** Early runs used a random 80/20 split per image.
  The same car often appears in several photos, so it ended up in both train
  and validation, and several classifiers reported 100% validation accuracy.
  The fix was to group photos per vehicle, which the final pipeline does with
  5-fold `StratifiedGroupKFold`.
- **Too many parameters for the data.** With 100–500 images per task,
  fine-tuning millions of parameters overfits easily and is noisy. A frozen
  DINOv2 backbone plus a linear layer (under 1,200 trainable parameters per
  task) trains in seconds, which makes cross-validation cheap.
- **BatchNorm bug.** Setting `requires_grad=False` does not freeze BatchNorm
  running statistics. They kept updating on tiny batches (4–16 images), which
  made the training curves noisy until the layers were explicitly put in
  `eval()` mode.
