"""Export ONNX model."""
from __future__ import annotations

import argparse
import torch
from torch import nn

from src.models.student_net import StudentNet


class ExportWrapper(nn.Module):
    def __init__(self, model: StudentNet):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        out = self.model(x)
        return out["logits"], out["mu_xy"], out["dir_xy"], out["l"], out["cov"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    m = StudentNet()
    m.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    m.eval()
    em = ExportWrapper(m)
    x = torch.randn(1, 9, 288, 512)
    torch.onnx.export(em, x, args.out, input_names=["x"], output_names=["logits", "mu_xy", "dir_xy", "l", "cov"], opset_version=17)


if __name__ == "__main__":
    main()
