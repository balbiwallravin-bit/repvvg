"""Train student KD model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.frame_window_dataset import FrameWindowDataset, kd_collate
from src.datasets.ready_index import parse_index
from src.losses.kd_losses import kd_total_loss
from src.models.student_net import StudentNet
from src.utils.io import append_jsonl, ensure_dir, write_lines
from src.utils.logging import setup_logger
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True)
    p.add_argument("--ready_root", required=True)
    p.add_argument("--pseudo_root", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default="")
    p.add_argument("--strict", type=int, default=0)
    p.add_argument("--log_every", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out = Path(args.out_dir)
    ckpt_dir = ensure_dir(out / "checkpoints")
    log_dir = ensure_dir(out / "logs")
    bad_dir = ensure_dir(out / "badcases")
    logger = setup_logger(log_dir / "train.log")

    specs = parse_index(args.index, args.ready_root)
    ds = FrameWindowDataset(specs, args.pseudo_root, strict=args.strict)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=kd_collate, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StudentNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp))
    start_epoch = 0
    best = 1e9
    ma = None

    if args.resume:
        st = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start_epoch = st.get("epoch", 0) + 1
        best = st.get("best", best)

    global_step = 0
    for ep in range(start_epoch, args.epochs):
        model.train()
        pbar = tqdm(dl, desc=f"epoch {ep}")
        for batch in pbar:
            if not batch:
                continue
            x = batch["x"].to(device, non_blocking=True)
            hm_t = batch["hm_t"].to(device, non_blocking=True)
            score = batch["score"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(args.amp)):
                out_d = model(x, return_logits=True, return_params=False)
                loss, d = kd_total_loss(out_d["logits"], hm_t, score, model.decoder)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            lv = float(d["loss"].item())
            ma = lv if ma is None else 0.9 * ma + 0.1 * lv
            if global_step % args.log_every == 0:
                rec = {
                    "step": global_step,
                    "epoch": ep,
                    "loss": lv,
                    "l_kl": float(d["l_kl"]),
                    "l_mu": float(d["l_mu"]),
                    "l_sigma": float(d["l_sigma"]),
                    "l_grad": float(d["l_grad"]),
                    "mu_err_px": float(d["mu_err_px"]),
                    "mean_score": float(batch["score"].mean().item()),
                    "mean_score_raw": float(batch["score_raw"].mean().item()),
                    "bad_npz_count": len(ds.badcases["missing_npz"]),
                }
                append_jsonl(out / "metrics.jsonl", rec)
                logger.info(json.dumps(rec, ensure_ascii=False))
            global_step += 1

        last_path = ckpt_dir / "last.pt"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep, "best": best}, last_path)
        if ma is not None and ma < best:
            best = ma
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep, "best": best}, ckpt_dir / "best.pt")

    write_lines(bad_dir / "missing_npz.txt", ds.badcases["missing_npz"])
    write_lines(bad_dir / "missing_frames.txt", ds.badcases["missing_frames"])
    total_bad = len(ds.badcases["missing_npz"]) + len(ds.badcases["missing_frames"])
    if len(ds) > 0 and total_bad / max(1, len(ds)) > 0.2:
        logger.warning("Bad sample ratio > 20%%")


if __name__ == "__main__":
    main()
