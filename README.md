# blurtrack_kd

Student KD training pipeline (Route A: Heatmap Distillation + Moments Decoder).

## Train

```bash
python -m src.train \
  --index /home/lht/blurtrack/video_maked/index_train_ready.jsonl \
  --ready_root /home/lht/blurtrack/video_maked_ready \
  --pseudo_root /home/lht/blurtrack/pseudo/heatmaps \
  --out_dir /home/lht/blurtrack/hello/blurtrack_kd/outputs/run1 \
  --batch_size 64 --epochs 50 --lr 3e-4 --amp 1 --num_workers 8
```

## Quick checks

```bash
python tools/sanity_check_dataset.py --index /home/lht/blurtrack/video_maked/index_train_ready.jsonl --ready_root /home/lht/blurtrack/video_maked_ready --pseudo_root /home/lht/blurtrack/pseudo/heatmaps
pytest -q
```


## Track new videos

```bash
python tools/track_videos.py   --videos /home/lht/ceshi/D1_S20251015112330_E20251015112400.mp4 /home/lht/ceshi/D1_S20251015112530_E20251015112553.mp4 /home/lht/ceshi/D1_S20251022153601_E20251022153631.mp4   --ckpt /home/lht/blurtrack/hello/blurtrack_kd/outputs/run1/checkpoints/best.pt   --out_root /home/lht/blurtrack/outputs   --log_every 30 \
  --roi_x0 367 --roi_y0 100 --roi_x1 1760 --roi_y1 884 \
  --smooth_alpha 0.35 --center_boost 1.0 \
  --roi_ref_w 1920 --roi_ref_h 1080 \
  --min_score_roi 0.12
```

Outputs are saved under `/home/lht/blurtrack/outputs/<start_timestamp>/` based on filename pattern like `_S20251015112530_`. The tracker rescales ROI from reference size (default 1920x1080) to each video resolution, applies ROI-masked heatmap decoding (track point constrained inside ROI), uses ROI-center-aware confidence (`score_roi`), and applies EMA smoothing for trajectory, and suppresses drawing in likely no-ball frames via `--min_score_roi`.


## Visualize training heatmaps

```bash
python tools/visualize_training_heatmaps.py \
  --index /home/lht/blurtrack/video_maked/index_train_ready.jsonl \
  --ready_root /home/lht/blurtrack/video_maked_ready \
  --pseudo_root /home/lht/blurtrack/pseudo/heatmaps \
  --out_dir /home/lht/blurtrack/outputs/heatmap_debug \
  --num_samples 30 \
  --mode random
```

Each image is `[t-1 | t | t+1 | raw_heatmap | overlay]`.


## Retrain with ROI-only + Visibility + Hard Negatives

```bash
python -m src.train   --index /home/lht/blurtrack/video_maked/index_train_ready.jsonl   --ready_root /home/lht/blurtrack/video_maked_ready   --pseudo_root /home/lht/blurtrack/pseudo/heatmaps   --out_dir /home/lht/blurtrack/outputs/run_roi_visi   --batch_size 64 --epochs 50 --lr 3e-4 --amp 1 --num_workers 8   --roi_enable 1 --roi_ref_w 1920 --roi_ref_h 1080   --roi_x0 367 --roi_y0 100 --roi_x1 1760 --roi_y1 884   --visi_thr 0.25 --hard_neg_ratio 0.2 --neg_hm_scale 0.1 --vis_loss_w 1.0
```

## Evaluate visible/no-ball split

```bash
python -m src.eval   --index /home/lht/blurtrack/video_maked/index_train_ready.jsonl   --ready_root /home/lht/blurtrack/video_maked_ready   --pseudo_root /home/lht/blurtrack/pseudo/heatmaps   --ckpt /home/lht/blurtrack/outputs/run_roi_visi   --batch_size 16 --max_batches 200 --log_every 20   --roi_enable 1 --roi_ref_w 1920 --roi_ref_h 1080   --roi_x0 367 --roi_y0 100 --roi_x1 1760 --roi_y1 884   --visi_thr 0.25 --vis_conf_thr 0.5
```

Eval output includes `fp_rate`, `fn_rate`, and `mu_err_px_visible`.


The `--ckpt` argument accepts either a checkpoint file or a run directory (auto-tries `checkpoints/best.pt` then `checkpoints/last.pt`).


If eval says checkpoints directory is empty, it means no `.pt` was produced yet (often training stopped too early). Re-run training and wait for at least one epoch end, or update to this version where `last.pt` is written at train start.


For multi-GPU training with `torchrun`, prefer script entry to avoid module-name collisions with external `src` packages:

```bash
CUDA_VISIBLE_DEVICES=4,5,6 torchrun --nproc_per_node=3 tools/train_torchrun.py \
  --index /home/lht/blurtrack/video_maked/index_train_ready.jsonl \
  --ready_root /home/lht/blurtrack/video_maked_ready \
  --pseudo_root /home/lht/blurtrack/pseudo/heatmaps \
  --out_dir /home/lht/blurtrack/outputs/run_roi_visi \
  --batch_size 64 --epochs 50 --lr 3e-4 --amp 1 --num_workers 8 \
  --roi_enable 1 --roi_ref_w 1920 --roi_ref_h 1080 \
  --roi_x0 367 --roi_y0 100 --roi_x1 1760 --roi_y1 884 \
  --visi_thr 0.25 --hard_neg_ratio 0.2 --neg_hm_scale 0.1 --vis_loss_w 1.0
```


When using torchrun, `--batch_size` is **per-process** (per GPU). With 3 GPUs, global batch is `3 * batch_size`. If OOM appears, reduce `--batch_size` (e.g. 16 or 8) first.
