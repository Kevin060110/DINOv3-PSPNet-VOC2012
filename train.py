from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dinov3_pspnet.data import make_loader
from dinov3_pspnet.engine import make_criterion, run_epoch
from dinov3_pspnet.utils import build_model, save_checkpoint, seed_everything, write_json


def parse_args():
    p = argparse.ArgumentParser(description="Train DINOv3 ViT-S/16 + PSPNet on VOC2012")
    p.add_argument("--dinov3-repo", required=True, help="cloned official DINOv3 repository")
    p.add_argument("--weights", required=True, help="authorized DINOv3 ViT-S/16 weight URL or local .pth")
    p.add_argument("--data-root", default="data")
    p.add_argument("--output-dir", default="outputs/frozen")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--download", action="store_true")
    p.add_argument("--finetune-backbone", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args).to(device)
    groups = [{"params": model.head.parameters(), "lr": args.lr}]
    if args.finetune_backbone:
        groups.append({"params": model.backbone.parameters(), "lr": args.backbone_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=args.epochs, power=0.9)
    criterion = make_criterion()
    train_loader = make_loader(args.data_root, "train", args.image_size, args.batch_size, args.workers, args.download)
    val_loader = make_loader(args.data_root, "val", args.image_size, args.batch_size, args.workers, args.download)
    history, best_miou = [], -1.0
    for epoch in range(1, args.epochs + 1):
        train_result = run_epoch(model, train_loader, device, criterion, optimizer, not args.no_amp)
        val_result = run_epoch(model, val_loader, device, criterion, amp=not args.no_amp)
        scheduler.step()
        row = {"epoch": epoch, "train": train_result, "val": val_result}
        history.append(row)
        print(f"epoch={epoch:03d} train_loss={train_result['loss']:.4f} "
              f"val_loss={val_result['loss']:.4f} val_mIoU={val_result['miou']:.4f}")
        if val_result["miou"] > best_miou:
            best_miou = val_result["miou"]
            save_checkpoint(Path(args.output_dir) / "best.pth", model, optimizer, epoch, best_miou, args)
        save_checkpoint(Path(args.output_dir) / "last.pth", model, optimizer, epoch, best_miou, args)
        write_json(Path(args.output_dir) / "history.json", history)


if __name__ == "__main__":
    main()

