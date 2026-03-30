"""
Lovasz fine-tuning: loads best seg model and fine-tunes with Lovasz loss.
Lovasz directly optimizes IoU. Expected: +2-3% IoU.
"""
import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2


def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits, labels):
    if len(labels) == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(torch.relu(errors_sorted), grad)
    return loss


def lovasz_hinge(logits, labels):
    loss = sum(
        lovasz_hinge_flat(log.reshape(-1), lab.reshape(-1))
        for log, lab in zip(logits, labels)
    )
    return loss / len(logits)


class CombinedLovaszLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode='binary')
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        return 0.3 * self.dice(logits, targets) + 0.2 * self.bce(logits, targets) + 0.5 * lovasz_hinge(logits, targets)


class SegDS(Dataset):
    def __init__(self, idir, mdir, tf=None):
        self.imgs = sorted(list(Path(idir).glob('*.jpg')) + list(Path(idir).glob('*.png')))
        self.mdir = Path(mdir)
        self.tf = tf

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        im = cv2.cvtColor(cv2.imread(str(self.imgs[i])), cv2.COLOR_BGR2RGB)
        mp = self.mdir / f'{self.imgs[i].stem}.png'
        if not mp.exists():
            mp = self.mdir / f'{self.imgs[i].stem}.jpg'
        m = cv2.imread(str(mp), 0) if mp.exists() else np.zeros((128, 128), np.uint8)
        m = (m > 127).astype(np.float32)
        if self.tf:
            r = self.tf(image=im, mask=m)
            im, m = r['image'], r['mask']
        return im, m.unsqueeze(0)


device = torch.device('cuda')

train_tf = A.Compose([
    A.Resize(384, 384), A.HorizontalFlip(0.5), A.VerticalFlip(0.5), A.RandomRotate90(0.5),
    A.ShiftScaleRotate(0.05, 0.1, 15, border_mode=0, p=0.3),
    A.RandomBrightnessContrast(0.15, 0.15, p=0.3),
    A.Normalize([.485, .456, .406], [.229, .224, .225]), ToTensorV2()
])
val_tf = A.Compose([
    A.Resize(384, 384), A.Normalize([.485, .456, .406], [.229, .224, .225]), ToTensorV2()
])

model = smp.UnetPlusPlus(encoder_name='tu-tf_efficientnetv2_s', encoder_weights=None, classes=1, activation=None).to(device)

ckpt = 'checkpoints/best_seg_384_effv2s.pth'
if os.path.exists(ckpt):
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
    print(f'Loaded {ckpt}')

tds = SegDS('data/Segmentation/training/images', 'data/Segmentation/training/masks', train_tf)
vds = SegDS('data/Segmentation/validation/images', 'data/Segmentation/validation/masks', val_tf)
print(f'Train:{len(tds)} Val:{len(vds)}')

tdl = DataLoader(tds, batch_size=12, shuffle=True, num_workers=8, pin_memory=True)
vdl = DataLoader(vds, batch_size=24, shuffle=False, num_workers=4, pin_memory=True)

criterion = CombinedLovaszLoss()
opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 20, 1e-6)

best = 0
os.makedirs('checkpoints', exist_ok=True)
print('Lovasz fine-tuning (20 epochs)...')

for e in range(20):
    model.train()
    for x, y in tdl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()
    sched.step()

    model.eval()
    ious = []
    with torch.no_grad():
        for x, y in vdl:
            x, y = x.to(device), y.to(device)
            p = (torch.sigmoid(model(x)) > 0.5).float()
            inter = (p * y).sum((2, 3))
            union = p.sum((2, 3)) + y.sum((2, 3)) - inter
            ious.extend(((inter + 1e-6) / (union + 1e-6)).squeeze().cpu().tolist())
    mi = np.mean(ious)
    if mi > best:
        best = mi
        torch.save(model.state_dict(), 'checkpoints/best_seg_lovasz_384.pth')
    if e % 2 == 0:
        print(f'  E{e}: IoU={mi:.4f} best={best:.4f}')

print(f'Lovasz DONE! Best={best:.4f}')
