"""
MEGA segmentation: 2000 train, 200 val, 512px, U-Net++ EfficientNetV2-S
Phase 1: Dice+BCE (40 epochs) -> Phase 2: Lovasz fine-tune (15 epochs)
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

def collect(idir, mdir):
    pairs = []
    for p in sorted(list(Path(idir).glob('*.jpg'))+list(Path(idir).glob('*.png'))):
        mp = Path(mdir)/f'{p.stem}.png'
        if not mp.exists(): mp = Path(mdir)/f'{p.stem}.jpg'
        if mp.exists(): pairs.append((str(p), str(mp)))
    return pairs

all_pairs = collect('data/Segmentation/training/images','data/Segmentation/training/masks') + \
            collect('data/Segmentation/validation/images','data/Segmentation/validation/masks')
print(f'Total: {len(all_pairs)}')
random.shuffle(all_pairs)
train_p, val_p = all_pairs[:-200], all_pairs[-200:]
print(f'Train:{len(train_p)} Val:{len(val_p)}')

ttf = A.Compose([A.Resize(512,512),A.HorizontalFlip(.5),A.VerticalFlip(.5),A.RandomRotate90(.5),
    A.ShiftScaleRotate(.1,.2,30,border_mode=0,p=.5),
    A.OneOf([A.HueSaturationValue(10,20,15,p=1),A.RandomBrightnessContrast(.2,.2,p=1)],p=.5),
    A.CLAHE(2.,p=.3),A.ElasticTransform(alpha=50,sigma=2.5,p=.2),
    A.Normalize([.485,.456,.406],[.229,.224,.225]),ToTensorV2()])
vtf = A.Compose([A.Resize(512,512),A.Normalize([.485,.456,.406],[.229,.224,.225]),ToTensorV2()])

d = torch.device('cuda')
model = smp.UnetPlusPlus(encoder_name='tu-tf_efficientnetv2_s',encoder_weights='imagenet',classes=1,activation=None).to(d)
tds = SegDS([p[0] for p in train_p],[p[1] for p in train_p],ttf)
vds = SegDS([p[0] for p in val_p],[p[1] for p in val_p],vtf)
tdl = DataLoader(tds,batch_size=8,shuffle=True,num_workers=8,pin_memory=True)
vdl = DataLoader(vds,batch_size=16,shuffle=False,num_workers=4,pin_memory=True)

dice_fn = smp.losses.DiceLoss(mode='binary'); bce_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
sc = torch.optim.lr_scheduler.CosineAnnealingLR(opt,40,1e-6)
best = 0; os.makedirs('checkpoints',exist_ok=True)
print('Phase1: Dice+BCE 512px 40ep')

for e in range(40):
    model.train()
    for x,y in tdl:
        x,y=x.to(d),y.to(d); opt.zero_grad()
        loss=.5*dice_fn(model(x),y)+.5*bce_fn(model(x),y); loss.backward(); opt.step()
    sc.step(); model.eval(); ious=[]
    with torch.no_grad():
        for x,y in vdl:
            x,y=x.to(d),y.to(d); p=(torch.sigmoid(model(x))>.5).float()
            i=(p*y).sum((2,3)); u=p.sum((2,3))+y.sum((2,3))-i
            ious.extend(((i+1e-6)/(u+1e-6)).squeeze().cpu().tolist())
    mi=np.mean(ious)
    if mi>best: best=mi; torch.save(model.state_dict(),'checkpoints/best_seg_mega_512.pth')
    if e%5==0: print(f'  P1 E{e}: IoU={mi:.4f} best={best:.4f}')
print(f'Phase1 DONE! Best={best:.4f}')

# Phase 2: Lovasz
def lovasz_grad(gs):
    p=len(gs); g=gs.sum(); i=g-gs.float().cumsum(0); u=g+(1-gs).float().cumsum(0)
    j=1.-i/u
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

model.load_state_dict(torch.load('checkpoints/best_seg_mega_512.pth',map_location=d))
opt2=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=.01)
sc2=torch.optim.lr_scheduler.CosineAnnealingLR(opt2,15,1e-6)
best2=best; print(f'Phase2: Lovasz 15ep from IoU={best:.4f}')

for e in range(15):
    model.train()
    for x,y in tdl:
        x,y=x.to(d),y.to(d); opt2.zero_grad()
        p=model(x); loss=.3*dice_fn(p,y)+.2*bce_fn(p,y)+.5*lovasz_hinge(p,y)
        loss.backward(); opt2.step()
    sc2.step(); model.eval(); ious=[]
    with torch.no_grad():
        for x,y in vdl:
            x,y=x.to(d),y.to(d); p=(torch.sigmoid(model(x))>.5).float()
            i=(p*y).sum((2,3)); u=p.sum((2,3))+y.sum((2,3))-i
            ious.extend(((i+1e-6)/(u+1e-6)).squeeze().cpu().tolist())
    mi=np.mean(ious)
    if mi>best2: best2=mi; torch.save(model.state_dict(),'checkpoints/best_seg_mega_512_lovasz.pth')
    if e%3==0: print(f'  P2 E{e}: IoU={mi:.4f} best={best2:.4f}')
print(f'MEGA DONE! P1={best:.4f} P2={best2:.4f}')
