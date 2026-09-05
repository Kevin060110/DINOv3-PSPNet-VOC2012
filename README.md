# DINOv3 ViT-S/16 + PSPNet on PASCAL VOC 2012

本项目实现题目要求的迁移学习语义分割流程：DINOv3 ViT-S/16 提取 patch token，转换成
`[B, 384, H/16, W/16]` 特征图，PSPNet 金字塔池化头输出 21 类 logits，再上采样至输入分辨率。

## 1. 环境与权重

```bash
python -m venv --system-site-packages .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
git clone https://github.com/facebookresearch/dinov3.git external/dinov3
```

DINOv3 官方权重需要先在官方页面申请访问权限。取得 ViT-S/16 的 `.pth` 文件或授权 URL 后，
通过 `--weights` 传入；`--dinov3-repo` 必须指向包含 `hubconf.py` 的官方仓库根目录。

## 2. 训练

先做冻结骨干的基线（推荐）：

```bash
python train.py --dinov3-repo external/dinov3 --weights checkpoints/dinov3_vits16.pth \
  --data-root data --download --output-dir outputs/frozen
```

再做端到端微调对照；骨干使用更小的学习率：

```bash
python train.py --dinov3-repo external/dinov3 --weights checkpoints/dinov3_vits16.pth \
  --data-root data --finetune-backbone --lr 1e-3 --backbone-lr 1e-5 \
  --output-dir outputs/finetuned
```

Windows PowerShell 中可把多行续行符 `\` 改成反引号，或把命令写在一行。显存不足时使用
`--image-size 384 --batch-size 2`。尺寸必须是 16 的倍数。

## 3. 独立验证

```bash
python evaluate.py --dinov3-repo external/dinov3 --weights checkpoints/dinov3_vits16.pth \
  --checkpoint outputs/frozen/best.pth --data-root data --output outputs/frozen/evaluation.json
```

程序报告验证 loss、pixel accuracy、mIoU 和 21 类 IoU。VOC mask 的 255 边界像素在 loss 与
混淆矩阵中均被忽略。训练结果保存在 `history.json`，最佳模型以验证集 mIoU 选择。

## 4. 实现说明与实验分析

- 数据：VOC2012 semantic segmentation 的 train/val split（不是 detection 标签），共 21 类，
  包含 background。图像双线性缩放，mask 用最近邻缩放，二者同步随机翻转。
- 骨干：官方 `dinov3_vits16`，patch size 16、embedding 384。使用
  `forward_features()["x_norm_patchtokens"]`，不使用 CLS/storage tokens。
- 分割头：对 1×1、2×2、3×3、6×6 四种空间尺度池化并拼接，随后卷积分类。
- 损失：逐像素交叉熵，`ignore_index=255`。
- 指标：先累积整个验证集的混淆矩阵，再计算每类 `TP/(TP+FP+FN)`；只对验证集中实际出现的
  类求均值，避免逐 batch 平均带来的偏差。
- 实验设计：冻结骨干用于检验预训练特征的线性/浅层可迁移性；端到端微调用低 100 倍的骨干
  学习率检验任务适配收益。两组必须使用相同数据划分、尺寸、epoch 和随机种子。

建议报告表格：

| 模型 | Backbone LR | Head LR | Val mIoU | Pixel Acc |
|---|---:|---:|---:|---:|
| Frozen DINOv3-S + PSP | 0 | 1e-3 | 实测填写 | 实测填写 |
| Fine-tuned DINOv3-S + PSP | 1e-5 | 1e-3 | 实测填写 | 实测填写 |

不能在未训练的情况下编造 mIoU。最终数值应从两次运行生成的 `evaluation.json` 抄录，并讨论：
小目标和细边界通常受 `/16` 特征步长与统一缩放影响；固定方形缩放会改变长宽比；VOC 官方
train 集规模较小，结果不应直接与使用额外增强数据、强训练策略或多尺度推理的论文数字比较。

## 5. 快速测试

测试不需要 DINOv3 权重：

```bash
python -m pytest -q
```

它验证 patch token 到二维特征图、输出尺寸恢复、骨干冻结和 ignore label 的 mIoU 计算。
