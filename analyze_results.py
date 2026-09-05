"""Generate training-curve figures from completed experiments."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).parent
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

histories = {
    name: json.loads((ROOT / "outputs" / name / "history.json").read_text(encoding="utf-8"))
    for name in ("frozen", "finetuned")
}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
colors = {"frozen": "#2563eb", "finetuned": "#dc2626"}
labels = {"frozen": "Frozen (384px)", "finetuned": "Fine-tuned (320px)"}

for name, history in histories.items():
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["train"]["miou"] * 100 for row in history], "--",
                 color=colors[name], alpha=0.65, label=f"{labels[name]} train")
    axes[0].plot(epochs, [row["val"]["miou"] * 100 for row in history],
                 color=colors[name], label=f"{labels[name]} val")
    axes[1].plot(epochs, [row["train"]["loss"] for row in history], "--",
                 color=colors[name], alpha=0.65, label=f"{labels[name]} train")
    axes[1].plot(epochs, [row["val"]["loss"] for row in history],
                 color=colors[name], label=f"{labels[name]} val")

axes[0].set(title="Training and validation mIoU", xlabel="Epoch", ylabel="mIoU (%)")
axes[1].set(title="Training and validation loss", xlabel="Epoch", ylabel="Cross-entropy loss")
for axis in axes:
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
fig.tight_layout()
fig.savefig(REPORT_DIR / "training_curves.png", dpi=180)
plt.close(fig)
