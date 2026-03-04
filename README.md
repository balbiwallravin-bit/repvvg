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
