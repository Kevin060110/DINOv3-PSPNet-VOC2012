from __future__ import annotations

import argparse
import torch

from dinov3_pspnet.data import make_loader
from dinov3_pspnet.engine import make_criterion, run_epoch
from dinov3_pspnet.utils import build_model, load_checkpoint, write_json


def main():
    p = argparse.ArgumentParser(description="Evaluate mIoU on VOC2012 val")
    p.add_argument("--dinov3-repo", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default="data")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output", default="outputs/evaluation.json")
    p.add_argument("--finetune-backbone", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args).to(device)
    load_checkpoint(args.checkpoint, model, device)
    loader = make_loader(args.data_root, "val", args.image_size, args.batch_size, args.workers)
    result = run_epoch(model, loader, device, make_criterion(), amp=not args.no_amp)
    write_json(args.output, result)
    print(f"loss={result['loss']:.4f} pixel_acc={result['pixel_accuracy']:.4f} mIoU={result['miou']:.4f}")


if __name__ == "__main__":
    main()

