from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        obj = json.loads(s)
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _linear_slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den > 0 else 0.0


def _metric_series(rows: list[dict[str, Any]], key: str) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    for i, r in enumerate(rows):
        if key in r:
            try:
                out.append((int(r.get("step", i)), float(r[key])))
            except Exception:
                continue
    return out


def _summarize_metric(rows: list[dict[str, Any]], key: str, lower_is_better: bool = True) -> dict[str, float] | None:
    s = _metric_series(rows, key)
    if not s:
        return None
    steps = [x for x, _ in s]
    vals = [y for _, y in s]
    first = vals[0]
    last = vals[-1]
    best = min(vals) if lower_is_better else max(vals)
    best_idx = vals.index(best)
    slope = _linear_slope([float(x) for x in steps], vals)
    return {
        "first": first,
        "last": last,
        "delta": last - first,
        "best": best,
        "best_step": float(steps[best_idx]),
        "mean": mean(vals),
        "slope_per_step": slope,
    }


def _epoch_summary(rows: list[dict[str, Any]], key: str) -> list[tuple[int, float]]:
    bucket: dict[int, list[float]] = {}
    for r in rows:
        if "epoch" not in r or key not in r:
            continue
        try:
            ep = int(r["epoch"])
            v = float(r[key])
        except Exception:
            continue
        bucket.setdefault(ep, []).append(v)
    return sorted((ep, mean(vals)) for ep, vals in bucket.items())


def build_report(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "# Training Report\n\nNo metrics found.\n"

    keys = [
        ("loss", True),
        ("l_kl", True),
        ("l_mu", True),
        ("l_sigma", True),
        ("l_vis", True),
        ("mu_err_px", True),
        ("mu_err_px_visible", True),
        ("visi_acc", False),
        ("mean_score", False),
        ("mean_visi", False),
    ]

    lines: list[str] = []
    lines.append("# Training Report")
    lines.append("")
    lines.append(f"- total_records: **{len(rows)}**")
    lines.append(f"- first_step: **{rows[0].get('step', 0)}**")
    lines.append(f"- last_step: **{rows[-1].get('step', len(rows)-1)}**")
    if "epoch" in rows[-1]:
        lines.append(f"- last_epoch: **{rows[-1]['epoch']}**")
    lines.append("")

    lines.append("## Key metric changes")
    lines.append("")
    lines.append("| metric | first | last | delta | best | best_step | mean | slope/step |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for key, lower_better in keys:
        s = _summarize_metric(rows, key, lower_is_better=lower_better)
        if s is None:
            continue
        lines.append(
            f"| {key} | {s['first']:.6f} | {s['last']:.6f} | {s['delta']:.6f} | {s['best']:.6f} | {int(s['best_step'])} | {s['mean']:.6f} | {s['slope_per_step']:.8f} |"
        )
    lines.append("")

    lines.append("## Epoch-level means")
    lines.append("")
    lines.append("| epoch | loss | mu_err_px_visible | visi_acc |")
    lines.append("|---:|---:|---:|---:|")
    by_loss = dict(_epoch_summary(rows, "loss"))
    by_mu = dict(_epoch_summary(rows, "mu_err_px_visible"))
    by_acc = dict(_epoch_summary(rows, "visi_acc"))
    all_eps = sorted(set(by_loss) | set(by_mu) | set(by_acc))
    for ep in all_eps:
        l = by_loss.get(ep, float("nan"))
        m = by_mu.get(ep, float("nan"))
        a = by_acc.get(ep, float("nan"))
        lines.append(f"| {ep} | {l:.6f} | {m:.6f} | {a:.6f} |")
    lines.append("")

    lines.append("## Latest snapshot")
    lines.append("")
    last = rows[-1]
    for k in ["step", "epoch", "loss", "l_kl", "l_mu", "l_sigma", "l_vis", "mu_err_px", "mu_err_px_visible", "visi_acc", "mean_score", "mean_visi"]:
        if k in last:
            lines.append(f"- {k}: `{last[k]}`")

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a training report from metrics.jsonl")
    ap.add_argument("--metrics", required=True, help="path to metrics.jsonl")
    ap.add_argument("--out_dir", required=True, help="output directory")
    ap.add_argument("--name", default="training_report", help="output basename")
    args = ap.parse_args()

    metrics_path = Path(args.metrics)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(metrics_path)
    report = build_report(rows)

    md_path = out_dir / f"{args.name}.md"
    json_path = out_dir / f"{args.name}.summary.json"

    md_path.write_text(report, encoding="utf-8")

    summary = {
        "records": len(rows),
        "metrics_path": str(metrics_path),
        "report_path": str(md_path),
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print({"report": str(md_path), "summary": str(json_path), "records": len(rows)})


if __name__ == "__main__":
    main()
