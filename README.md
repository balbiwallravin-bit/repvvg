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
python tools/track_videos.py   --videos /home/lht/ceshi/D1_S20251015112330_E20251015112400.mp4 /home/lht/ceshi/D1_S20251015112530_E20251015112553.mp4 /home/lht/ceshi/D1_S20251022153601_E20251022153631.mp4   --ckpt /home/lht/blurtrack/hello/blurtrack_kd/outputs/run1/checkpoints/best.pt   --out_root /home/lht/blurtrack/outputs   --log_every 30
```

Outputs are saved under `/home/lht/blurtrack/outputs/<start_timestamp>/` based on filename pattern like `_S20251015112530_`.
