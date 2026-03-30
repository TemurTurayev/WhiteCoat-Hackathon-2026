"""
ISIC 2018 Pretrain -> Hackathon Fine-tune for Segmentation.
Phase 1: Pretrain on ISIC 2018 (2594 masks) - 20 epochs
Phase 2: Fine-tune on hackathon data (2000 imgs) - 40 epochs
Phase 3: Lovasz fine-tune - 15 epochs
Model sees ~4600 unique skin lesion masks total!
"""
import torch, torch.nn as nn, segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2, numpy as np, os, random
from pathlib import Path

random.seed(42); np.random.seed(42); torch.manual_seed(42)

class SegDS(Dataset):
    def __init__(self, img_paths, mask_paths, tf=None):
        self.imgs = img_paths; self.masks = mask_paths; self.tf = tf
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        im = cv2.cvtColor(cv2.imread(str(self.imgs[i])), cv2.COLOR_BGR2RGB)
        m = cv2.imread(str(self.masks[i]), 0)
        m = np.zeros((128,128), np.uint8) if m is None else m
        m = (m > 127).astype(np.float32)
        if self.tf: r = self.tf(image=im, mask=m); im, m = r['image'], r['mask']
        return im, m.unsqueeze(0)

def get_tf(sz, train=True):
    if train:
        return A.Compose([A.Resize(sz,sz),A.HorizontalFlip(.5),A.VerticalFlip(.5),A.RandomRotate90(.5),
            A.ShiftScaleRotate(.1,.2,30,border_mode=0,p=.5),
            A.OneOf([A.HueSaturationValue(10,20,15,p=1),A.RandomBrightnessContrast(.2,.2,p=1)],p=.5),
            A.CLAHE(2.,p=.3),A.Normalize([.485,.456,.406],[.229,.224,.225]),ToTensorV2()])
    return A.Compose([A.Resize(sz,sz),A.Normalize([.485,.456,.406],[.229,.224,.225]),ToTensorV2()])

def do_train(model, tdl, vdl, epochs, lr, name, device, save_path):
    dice_fn = smp.losses.DiceLoss(mode='binary'); bce_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
    sc = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, 1e-6)
    best = 0
    for e in range(epochs):
        model.train()
        for x,y in tdl:
            x,y = x.to(device),y.to(device); opt.zero_grad()
            p = model(x); l = .5*dice_fn(p,y)+.5*bce_fn(p,y); l.backward(); opt.step()
        sc.step(); model.eval(); ious = []
        with torch.no_grad():
            for x,y in vdl:
                x,y = x.to(device),y.to(device)
                p = (torch.sigmoid(model(x))>.5).float()
                ii = (p*y).sum((2,3)); u = p.sum((2,3))+y.sum((2,3))-ii
                ious.extend(((ii+1e-6)/(u+1e-6)).squeeze().cpu().tolist())
        mi = np.mean(ious)
        if mi > best: best = mi; torch.save(model.state_dict(), save_path)
        if e % 5 == 0: print(f'  {name} E{e}: IoU={mi:.4f} best={best:.4f}')
    print(f'{name} DONE! Best={best:.4f}')
    return best

device = torch.device('cuda')
os.makedirs('checkpoints', exist_ok=True)
SZ = 384

# PHASE 1: ISIC 2018 PRETRAIN
print('='*50)
print('PHASE 1: ISIC 2018 Pretrain')

isic_idir = Path('isic_seg/images_raw/ISIC2018_Task1-2_Training_Input')
isic_mdir = Path('isic_seg/masks_raw/ISIC2018_Task1_Training_GroundTruth')
isic_pairs = []
for ip in sorted(isic_idir.glob('*.jpg')):
    mp = isic_mdir / f'{ip.stem}_segmentation.png'
    if mp.exists(): isic_pairs.append((str(ip), str(mp)))
print(f'ISIC pairs: {len(isic_pairs)}')
random.shuffle(isic_pairs)
itr, ivl = isic_pairs[:-200], isic_pairs[-200:]

model = smp.UnetPlusPlus(encoder_name='tu-tf_efficientnetv2_s', encoder_weights='imagenet', classes=1, activation=None).to(device)
itdl = DataLoader(SegDS([p[0] for p in itr],[p[1] for p in itr],get_tf(SZ,True)),batch_size=12,shuffle=True,num_workers=8,pin_memory=True)
ivdl = DataLoader(SegDS([p[0] for p in ivl],[p[1] for p in ivl],get_tf(SZ,False)),batch_size=24,shuffle=False,num_workers=4,pin_memory=True)
ib = do_train(model, itdl, ivdl, 20, 3e-4, 'ISIC', device, 'checkpoints/seg_isic_pretrain.pth')

# PHASE 2: HACKATHON FINE-TUNE
print('='*50)
print('PHASE 2: Hackathon Fine-tune')

def collect(idir, mdir):
    pairs = []
    for p in sorted(list(Path(idir).glob('*.jpg'))+list(Path(idir).glob('*.png'))):
        mp = Path(mdir)/f'{p.stem}.png'
        if not mp.exists(): mp = Path(mdir)/f'{p.stem}.jpg'
        if mp.exists(): pairs.append((str(p), str(mp)))
    return pairs

hp = collect('data/Segmentation/training/images','data/Segmentation/training/masks') + \
     collect('data/Segmentation/validation/images','data/Segmentation/validation/masks')
print(f'Hackathon pairs: {len(hp)}')
random.shuffle(hp)
htr, hvl = hp[:-200], hp[-200:]

model.load_state_dict(torch.load('checkpoints/seg_isic_pretrain.pth', map_location=device))
htdl = DataLoader(SegDS([p[0] for p in htr],[p[1] for p in htr],get_tf(SZ,True)),batch_size=12,shuffle=True,num_workers=8,pin_memory=True)
hvdl = DataLoader(SegDS([p[0] for p in hvl],[p[1] for p in hvl],get_tf(SZ,False)),batch_size=24,shuffle=False,num_workers=4,pin_memory=True)
hb = do_train(model, htdl, hvdl, 40, 1e-4, 'Hack', device, 'checkpoints/seg_isic_then_hack.pth')

# PHASE 3: LOVASZ
print('='*50)
print('PHASE 3: Lovasz')

def lovasz_grad(gs):
    p=len(gs); g=gs.sum(); ii=g-gs.float().cumsum(0); u=g+(1-gs).float().cumsum(0)
    j=1.-ii/u
    if p>1: j[1:]=j[1:]-j[:-1]
    return j
def lovasz_hinge(logits, labels):
    loss=0
    for lo,la in zip(logits,labels):
        lf,lbf=lo.reshape(-1),la.reshape(-1)
        if len(lbf)==0: continue
        err=1.-(2.*lbf.float()-1.)*lf
        es,pm=torch.sort(err,descending=True)
        loss+=torch.dot(torch.relu(es),lovasz_grad(lbf[pm]))
    return loss/max(len(logits),1)

model.load_state_dict(torch.load('checkpoints/seg_isic_then_hack.pth', map_location=device))
dice_fn = smp.losses.DiceLoss(mode='binary'); bce_fn = nn.BCEWithLogitsLoss()
o3 = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=.01)
s3 = torch.optim.lr_scheduler.CosineAnnealingLR(o3, 15, 1e-6)
b3 = hb

for e in range(15):
    model.train()
    for x,y in htdl:
        x,y = x.to(device),y.to(device); o3.zero_grad()
        p = model(x); l = .3*dice_fn(p,y)+.2*bce_fn(p,y)+.5*lovasz_hinge(p,y)
        l.backward(); o3.step()
    s3.step(); model.eval(); ious = []
    with torch.no_grad():
        for x,y in hvdl:
            x,y = x.to(device),y.to(device)
            p = (torch.sigmoid(model(x))>.5).float()
            ii = (p*y).sum((2,3)); u = p.sum((2,3))+y.sum((2,3))-ii
            ious.extend(((ii+1e-6)/(u+1e-6)).squeeze().cpu().tolist())
    mi = np.mean(ious)
    if mi > b3: b3 = mi; torch.save(model.state_dict(), 'checkpoints/seg_isic_hack_lovasz.pth')
    if e % 3 == 0: print(f'  Lovasz E{e}: IoU={mi:.4f} best={b3:.4f}')

print(f'FINAL: ISIC={ib:.4f} -> Hack={hb:.4f} -> Lovasz={b3:.4f}')
