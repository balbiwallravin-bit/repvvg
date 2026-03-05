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
    p.add_argument("--batch_size", type=int, default=64, help="per-process batch size when using torchrun")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default="")
    p.add_argument("--strict", type=int, default=0)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--devices", default="", help="for non-torchrun single-process DataParallel only")

    p.add_argument("--roi_enable", type=int, default=1)
    p.add_argument("--roi_ref_w", type=int, default=1920)
    p.add_argument("--roi_ref_h", type=int, default=1080)
    p.add_argument("--roi_x0", type=int, default=367)
    p.add_argument("--roi_y0", type=int, default=100)
    p.add_argument("--roi_x1", type=int, default=1760)
    p.add_argument("--roi_y1", type=int, default=884)

    p.add_argument("--visi_thr", type=float, default=0.25)
    p.add_argument("--hard_neg_ratio", type=float, default=0.2)
    p.add_argument("--neg_hm_scale", type=float, default=0.1)
    p.add_argument("--vis_loss_w", type=float, default=1.0)

    p.add_argument("--local_rank", type=int, default=0)
    p.add_argument("--local-rank", dest="local_rank", type=int, default=0)

    args, unknown = p.parse_known_args()
    if unknown:
        print({"ignored_unknown_args": unknown})
    return args


def main() -> None:
    args = parse_args()
    print({"train_entry": __file__})

    use_cuda = torch.cuda.is_available()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    rank = int(os.environ.get("RANK", "0"))

    if distributed:
        backend = "nccl" if use_cuda else "gloo"
        dist.init_process_group(backend=backend)

    set_seed(args.seed + rank)
    out = Path(args.out_dir)
    ckpt_dir = ensure_dir(out / "checkpoints")
    log_dir = ensure_dir(out / "logs")
    bad_dir = ensure_dir(out / "badcases")
    logger = setup_logger(log_dir / "train.log")

    specs = parse_index(args.index, args.ready_root)
    if rank == 0:
        logger.info(f"parsed_samples={len(specs)} from index={args.index}")
    if len(specs) == 0:
        raise RuntimeError("No valid samples parsed from index file.")

    ds = FrameWindowDataset(
        specs,
        args.pseudo_root,
        strict=args.strict,
        roi_enable=args.roi_enable,
        roi_ref_w=args.roi_ref_w,
        roi_ref_h=args.roi_ref_h,
        roi_x0=args.roi_x0,
        roi_y0=args.roi_y0,
        roi_x1=args.roi_x1,
        roi_y1=args.roi_y1,
        visi_thr=args.visi_thr,
        hard_neg_ratio=args.hard_neg_ratio,
    )

    if distributed:
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=kd_collate,
        pin_memory=True,
    )

    if use_cuda:
        if distributed:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            if args.devices:
                dev_ids = [int(x) for x in args.devices.split(",") if x.strip() != ""]
            else:
                dev_ids = list(range(torch.cuda.device_count()))
            dev_ids = dev_ids or [0]
            torch.cuda.set_device(dev_ids[0])
            device = torch.device(f"cuda:{dev_ids[0]}")
    else:
        device = torch.device("cpu")

    model = StudentNet().to(device)

    if distributed and use_cuda:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        if rank == 0:
            logger.info(f"Using DistributedDataParallel world_size={world_size}")
    elif use_cuda:
        if args.devices:
            dev_ids = [int(x) for x in args.devices.split(",") if x.strip() != ""]
        else:
            dev_ids = list(range(torch.cuda.device_count()))
        if len(dev_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=dev_ids)
            logger.info(f"Using DataParallel on GPUs: {dev_ids}")
        else:
            logger.info(f"Using single GPU: {device}")
    else:
        logger.info("Using CPU")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and use_cuda))
    start_epoch = 0
    best = 1e9
    ma = None

    if args.resume:
        st = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(st["model"]) if not isinstance(model, (torch.nn.DataParallel, DDP)) else model.module.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start_epoch = st.get("epoch", 0) + 1
        best = st.get("best", best)

    if rank == 0:
        init_state = model.module.state_dict() if isinstance(model, (torch.nn.DataParallel, DDP)) else model.state_dict()
        torch.save({"model": init_state, "opt": opt.state_dict(), "epoch": start_epoch - 1, "best": best}, ckpt_dir / "last.pt")

    global_step = 0
    for ep in range(start_epoch, args.epochs):
        model.train()
        if sampler is not None:
            sampler.set_epoch(ep)
        pbar = tqdm(dl, desc=f"epoch {ep}", disable=(rank != 0))

        for batch in pbar:
            if not batch:
                continue
            x = batch["x"].to(device, non_blocking=True)
            hm_t = batch["hm_t"].to(device, non_blocking=True)
            score = batch["score"].to(device, non_blocking=True)
            visi = batch["visi"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            amp_ctx = torch.amp.autocast(device_type="cuda", enabled=True) if (args.amp and use_cuda) else nullcontext()
            with amp_ctx:
                out_d = model(x, return_logits=True, return_params=False)
                decoder = model.module.decoder if isinstance(model, (torch.nn.DataParallel, DDP)) else model.decoder
                loss, d = kd_total_loss(
                    out_d["logits"],
                    hm_t,
                    score,
                    decoder,
                    visi_t=visi,
                    visi_logit_s=out_d["visi_logit"],
                    neg_hm_scale=args.neg_hm_scale,
                    e=args.vis_loss_w,
                )
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            lv = float(d["loss"].item())
            ma = lv if ma is None else 0.9 * ma + 0.1 * lv
            if rank == 0 and global_step % args.log_every == 0:
                rec = {
                    "step": global_step,
                    "epoch": ep,
                    "loss": lv,
                    "l_kl": float(d["l_kl"]),
                    "l_mu": float(d["l_mu"]),
                    "l_sigma": float(d["l_sigma"]),
                    "l_grad": float(d["l_grad"]),
                    "l_vis": float(d["l_vis"]),
                    "visi_acc": float(d["visi_acc"]),
                    "mu_err_px": float(d["mu_err_px"]),
                    "mu_err_px_visible": float(d["mu_err_px_visible"]),
                    "mean_score": float(batch["score"].mean().item()),
                    "mean_visi": float(batch["visi"].mean().item()),
                    "mean_score_raw": float(torch.nan_to_num(batch["score_raw"], nan=0.0).mean().item()),
                    "bad_npz_count": len(ds.badcases["missing_npz"]),
                }
                append_jsonl(out / "metrics.jsonl", rec)
                logger.info(json.dumps(rec, ensure_ascii=False))
            global_step += 1

        if rank == 0:
            last_path = ckpt_dir / "last.pt"
            model_state = model.module.state_dict() if isinstance(model, (torch.nn.DataParallel, DDP)) else model.state_dict()
            torch.save({"model": model_state, "opt": opt.state_dict(), "epoch": ep, "best": best}, last_path)
            if ma is not None and ma < best:
                best = ma
                torch.save({"model": model_state, "opt": opt.state_dict(), "epoch": ep, "best": best}, ckpt_dir / "best.pt")

    if rank == 0:
        write_lines(bad_dir / "missing_npz.txt", ds.badcases["missing_npz"])
        write_lines(bad_dir / "missing_frames.txt", ds.badcases["missing_frames"])

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
