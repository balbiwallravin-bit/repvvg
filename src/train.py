"""Train student KD model."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
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
    p.add_argument("--devices", default="", help="comma-separated cuda device ids, e.g. 0,1,2,3")
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
    logger.info(f"parsed_samples={len(specs)} from index={args.index}")
    if len(specs) == 0:
        raise RuntimeError(
            "No valid samples parsed from index file. "
            "Please verify index schema and segment paths."
        )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    use_cuda = torch.cuda.is_available()
    if use_cuda and args.devices and not distributed:
        dev_ids = [int(x) for x in args.devices.split(",") if x.strip() != ""]
        if not dev_ids:
            dev_ids = list(range(torch.cuda.device_count()))
    else:
        dev_ids = list(range(torch.cuda.device_count())) if use_cuda else []

    if use_cuda and distributed:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    elif use_cuda and dev_ids:
        torch.cuda.set_device(dev_ids[0])
        device = torch.device(f"cuda:{dev_ids[0]}")
    else:
        device = torch.device("cpu")

    model = StudentNet().to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank] if use_cuda else None)
        logger.info(f"Using DistributedDataParallel rank={rank}/{world_size} local_rank={local_rank}")
    elif use_cuda and len(dev_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=dev_ids)
        logger.info(f"Using DataParallel on GPUs: {dev_ids}")
    elif use_cuda and len(dev_ids) == 1:
        logger.info(f"Using single GPU: {dev_ids[0]}")
    else:
        logger.info("Using CPU")

    ds = FrameWindowDataset(specs, args.pseudo_root, strict=args.strict)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=kd_collate,
        pin_memory=True,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and use_cuda))
    start_epoch = 0
    best = 1e9
    ma = None

    if args.resume:
        st = torch.load(args.resume, map_location="cpu")
        if isinstance(model, (torch.nn.DataParallel, DDP)):
            model.module.load_state_dict(st["model"])
        else:
            model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start_epoch = st.get("epoch", 0) + 1
        best = st.get("best", best)

    global_step = 0
    for ep in range(start_epoch, args.epochs):
        model.train()
        if sampler is not None:
            sampler.set_epoch(ep)
        pbar = tqdm(dl, desc=f"epoch {ep}", disable=distributed and rank != 0)
        for batch in pbar:
            if not batch:
                continue
            x = batch["x"].to(device, non_blocking=True)
            hm_t = batch["hm_t"].to(device, non_blocking=True)
            score = batch["score"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            amp_ctx = torch.amp.autocast(device_type="cuda", enabled=True) if (args.amp and use_cuda) else nullcontext()
            with amp_ctx:
                out_d = model(x, return_logits=True, return_params=False)
                decoder = model.module.decoder if isinstance(model, (torch.nn.DataParallel, DDP)) else model.decoder
                loss, d = kd_total_loss(out_d["logits"], hm_t, score, decoder)
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
                    "mean_score_raw": float(torch.nan_to_num(batch["score_raw"], nan=0.0).mean().item()),
                    "bad_npz_count": len(ds.badcases["missing_npz"]),
                }
                if not distributed or rank == 0:
                    append_jsonl(out / "metrics.jsonl", rec)
                    logger.info(json.dumps(rec, ensure_ascii=False))
            global_step += 1

        if not distributed or rank == 0:
            last_path = ckpt_dir / "last.pt"
            model_state = model.module.state_dict() if isinstance(model, (torch.nn.DataParallel, DDP)) else model.state_dict()
            torch.save({"model": model_state, "opt": opt.state_dict(), "epoch": ep, "best": best}, last_path)
            if ma is not None and ma < best:
                best = ma
                torch.save({"model": model_state, "opt": opt.state_dict(), "epoch": ep, "best": best}, ckpt_dir / "best.pt")

    if not distributed or rank == 0:
        write_lines(bad_dir / "missing_npz.txt", ds.badcases["missing_npz"])
        write_lines(bad_dir / "missing_frames.txt", ds.badcases["missing_frames"])
    total_bad = len(ds.badcases["missing_npz"]) + len(ds.badcases["missing_frames"])
    if (not distributed or rank == 0) and len(ds) > 0 and total_bad / max(1, len(ds)) > 0.2:
        logger.warning("Bad sample ratio > 20%%")

    if distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
