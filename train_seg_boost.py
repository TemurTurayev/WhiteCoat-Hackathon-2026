import torch, torch.nn as nn, segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2, numpy as np, os
from pathlib import Path

class SegDS(Dataset):
    def __init__(self, idir, mdir, tf=None):
        self.imgs = sorted(list(Path(idir).glob('*.jpg'))+list(Path(idir).glob('*.png')))
        self.mdir = Path(mdir)
        self.tf = tf
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, i):
        im = cv2.cvtColor(cv2.imread(str(self.imgs[i])), cv2.COLOR_BGR2RGB)
        mp = self.mdir / f'{self.imgs[i].stem}.png'
        if not mp.exists():
            mp = self.mdir / f'{self.imgs[i].stem}.jpg'
        m = cv2.imread(str(mp), 0) if mp.exists() else np.zeros((128,128), np.uint8)
        m = (m > 127).astype(np.float32)
        if self.tf:
            r = self.tf(image=im, mask=m)
            im, m = r['image'], r['mask']
        return im, m.unsqueeze(0)

ttf = A.Compose([
    A.Resize(384,384), A.HorizontalFlip(0.5), A.VerticalFlip(0.5),
    A.RandomRotate90(0.5), A.ShiftScaleRotate(0.1,0.2,30,border_mode=0,p=0.5),
    A.RandomBrightnessContrast(0.2,0.2,p=0.5), A.CLAHE(2.0,p=0.3),
    A.Normalize([.485,.456,.406],[.229,.224,.225]), ToTensorV2()
])
vtf = A.Compose([
    A.Resize(384,384), A.Normalize([.485,.456,.406],[.229,.224,.225]), ToTensorV2()
])

d = torch.device('cuda')
m = smp.UnetPlusPlus(
    encoder_name='tu-tf_efficientnetv2_s',
    encoder_weights='imagenet', classes=1, activation=None
).to(d)

tds = SegDS('data/Segmentation/training/images','data/Segmentation/training/masks', ttf)
vds = SegDS('data/Segmentation/validation/images','data/Segmentation/validation/masks', vtf)
print(f'Train:{len(tds)} Val:{len(vds)}')

tdl = DataLoader(tds, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)
vdl = DataLoader(vds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

o = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
sc = torch.optim.lr_scheduler.CosineAnnealingLR(o, 60, 1e-6)
dl = smp.losses.DiceLoss(mode='binary')
bl = nn.BCEWithLogitsLoss()

best = 0
os.makedirs('checkpoints', exist_ok=True)

for e in range(60):
    m.train()
    for x,y in tdl:
        x,y = x.to(d), y.to(d)
        o.zero_grad()
        p = m(x)
        l = 0.5*dl(p,y) + 0.5*bl(p,y)
        l.backward()
        o.step()
    sc.step()
    m.eval()
    ious = []
    with torch.no_grad():
        for x,y in vdl:
            x,y = x.to(d), y.to(d)
            p = (torch.sigmoid(m(x)) > 0.5).float()
            inter = (p*y).sum((2,3))
            union = p.sum((2,3)) + y.sum((2,3)) - inter
            iou = (inter+1e-6) / (union+1e-6)
            ious.extend(iou.squeeze().cpu().tolist())
    mi = np.mean(ious)
    if mi > best:
        best = mi
        torch.save(m.state_dict(), 'checkpoints/best_seg_384_effv2s.pth')
    if e % 5 == 0:
        print(f'E{e}: IoU={mi:.4f} best={best:.4f}')
print(f'DONE! Best={best:.4f}')
