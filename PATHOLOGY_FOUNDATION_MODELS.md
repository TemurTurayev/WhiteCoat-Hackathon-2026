# Pathology Foundation Models -- Deep Research for Hackathon

## Executive Summary & Recommendation

**For the hackathon, use Phikon-v2 (Owkin) as your primary model.** It is:
- Freely downloadable (no institutional email required)
- The largest publicly available pathology foundation model (ViT-L, 300M params)
- Loads in 5 lines via HuggingFace `transformers`
- Proven state-of-the-art on tissue classification benchmarks
- Suitable for both classification (feature extraction + linear head) and segmentation (via torchseg)

**Runner-up: CTransPath** -- smaller (27.5M params), faster, Swin Transformer, loads via `timm`, good benchmarks, freely available via HuggingFace.

---

## 1. Phikon / Phikon-v2 (Owkin) -- TOP RECOMMENDATION

### URLs
- HuggingFace (v2): https://huggingface.co/owkin/phikon-v2
- HuggingFace (v1): https://huggingface.co/owkin/phikon
- GitHub: https://github.com/owkin/HistoSSLscaling
- Paper: https://arxiv.org/abs/2409.09173

### Specifications
| Attribute | Phikon (v1) | Phikon-v2 |
|-----------|-------------|-----------|
| Architecture | ViT-B/16 | ViT-L/16 |
| Parameters | 85.8M | 300M |
| Training method | iBOT | DINOv2 (DINO + iBOT + KoLeo) |
| Training data | 40M tiles (TCGA) | 456M tiles (TCGA, CPTAC, GTEx, TCIA) |
| Output dim | 768 | 1024 |
| Expected input | 224x224 | 224x224 |
| License | Owkin non-commercial | Owkin non-commercial |
| Download size | ~330MB | ~1.2GB |
| Institutional email? | NO | NO |

### Load and Extract Features (5 lines)
```python
from transformers import AutoImageProcessor, AutoModel
import torch

processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
model = AutoModel.from_pretrained("owkin/phikon-v2")
model.set_grad_enabled(False)

# Extract features from a PIL image
inputs = processor(image, return_tensors="pt")
with torch.inference_mode():
    features = model(**inputs).last_hidden_state[:, 0, :]  # shape: (1, 1024)
```

### Add Classification Head (12 classes)
```python
import torch.nn as nn

class PhikonClassifier(nn.Module):
    def __init__(self, num_classes=12, freeze_backbone=True):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
        self.backbone = AutoModel.from_pretrained("owkin/phikon-v2")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, pixel_values):
        with torch.no_grad():
            outputs = self.backbone(pixel_values=pixel_values)
            features = outputs.last_hidden_state[:, 0, :]  # CLS token
        return self.classifier(features)
```

### Handling 100x100 Images
Phikon-v2 expects 224x224. The `AutoImageProcessor` will automatically resize your 100x100 images to 224x224 using bilinear interpolation. This is fine -- the model handles it gracefully. You can also:

```python
# Option 1: Let processor handle it (default -- recommended)
inputs = processor(image_100x100, return_tensors="pt")

# Option 2: Manual resize with padding to preserve aspect ratio
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### Benchmark Results (from paper)
- Performs on par with models trained on proprietary data (UNI, Virchow)
- Camelyon16 (tumor detection): competitive with UNI
- TCGA tissue classification: strong linear probe accuracy
- Outperforms CTransPath across most benchmarks

### Can it be used as encoder for SMP segmentation?
**Not directly with SMP**, but YES with `torchseg` (see Section 9 below). ViT models do not produce multi-scale feature maps natively, so standard SMP cannot use them. But torchseg solves this.

---

## 2. UNI / UNI2 (MahmoodLab, Harvard)

### URLs
- HuggingFace (UNI): https://huggingface.co/MahmoodLab/UNI
- HuggingFace (UNI2-h): https://huggingface.co/MahmoodLab/UNI2-h
- GitHub: https://github.com/mahmoodlab/UNI
- Paper: https://www.nature.com/articles/s41591-024-02857-3

### Specifications
| Attribute | UNI | UNI2-h |
|-----------|-----|--------|
| Architecture | ViT-L/16 | ViT-H/14 |
| Parameters | 307M | 632M |
| Training data | 100M images (100K WSIs) | 200M images (350K WSIs) |
| Output dim | 1024 | 1280 |
| License | CC-BY-NC-ND 4.0 | CC-BY-NC-ND 4.0 |

### ACCESS REQUIREMENTS -- PROBLEM
- **Requires institutional email** (no @gmail, @hotmail, @qq)
- Must fill out a gated form with full name, affiliation, research description
- Personal email requests are automatically DENIED
- Each team member must register individually
- Commercial use requires contacting authors

### Workaround if you cannot get access
1. Use Phikon-v2 instead -- comparable performance, no access gate
2. Try using a university email (TashPMI email if available)
3. Check if KatherLab mirror exists: https://github.com/KatherLab/uni

### Loading code (if you have access)
```python
import timm
model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True)
model = model.eval()
```

### Performance
- Highest win rate in benchmark comparisons (0.64 win rate vs Phikon)
- Best on 10+ datasets for tissue classification
- Segmentation: 0.827 dice for epithelial, 0.690 smooth muscle, 0.803 RBC

---

## 3. Virchow / Virchow2 (Paige AI)

### URLs
- HuggingFace (Virchow): https://huggingface.co/paige-ai/Virchow
- HuggingFace (Virchow2): https://huggingface.co/paige-ai/Virchow2
- Paper: https://www.nature.com/articles/s41591-024-03141-0

### Specifications
| Attribute | Virchow | Virchow2 | Virchow2G |
|-----------|---------|----------|-----------|
| Architecture | ViT-H/14 | ViT-H/14 | ViT-G |
| Parameters | 632M | 632M | 1.85B |
| Training data | 1.5M WSIs | 3.1M WSIs | 3.1M WSIs |
| License | CC-BY-NC-ND 4.0 | CC-BY-NC-ND 4.0 | CC-BY-NC-ND 4.0 |

### ACCESS REQUIREMENTS
- Requires HuggingFace registration and agreement to terms
- Non-commercial research only
- **Less restrictive than UNI** -- does not explicitly require institutional email
- But gated model, needs approval

### Loading code
```python
import timm
model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True)
model = model.eval()
```

### Comparison
- Trained on the largest dataset (3.1M WSIs for Virchow2)
- Virchow2G is the largest pathology model at 1.85B params
- Too large for hackathon use (slow inference, high memory)
- **Verdict: overkill for a hackathon; use Phikon-v2 instead**

---

## 4. CONCH (MahmoodLab) -- Vision-Language Model

### URLs
- HuggingFace: https://huggingface.co/MahmoodLab/CONCH
- GitHub: https://github.com/mahmoodlab/CONCH
- Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC11384335/

### Specifications
| Attribute | Value |
|-----------|-------|
| Architecture | ViT-B + CoCa (vision-language) |
| Training data | 1.17M image-caption pairs |
| License | CC-BY-NC-ND 4.0 |
| Unique feature | Zero-shot classification via text prompts |

### ACCESS REQUIREMENTS -- PROBLEM
- **Requires institutional email** (same as UNI)
- Gated form with affiliation verification
- Personal email requests DENIED
- Academic use only

### Zero-Shot Classification (could identify your 12 classes)
```python
# If you have access:
from conch.open_clip_custom import create_model_from_pretrained

model, preprocess = create_model_from_pretrained(
    "conch_ViT-B-16", "hf_hub:MahmoodLab/CONCH"
)

# Zero-shot: describe classes in text
text_prompts = [
    "adenocarcinoma tissue",
    "squamous cell carcinoma",
    "normal epithelial tissue",
    # ... your 12 classes
]
```

### Why it matters for us
- If you can get access, use it to VERIFY what your 12 tissue classes are
- Zero-shot can help with initial data exploration
- **But for actual classification, Phikon-v2 feature extraction + linear head will be faster and more accurate with labeled data**

---

## 5. CTransPath -- SECOND RECOMMENDATION

### URLs
- HuggingFace (timm): https://huggingface.co/1aurent/swin_tiny_patch4_window7_224.CTransPath
- HuggingFace (alt): https://huggingface.co/jamesdolezal/CTransPath
- GitHub: https://github.com/Xiyue-Wang/TransPath
- Google Drive (original): https://drive.google.com/file/d/1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX/view

### Specifications
| Attribute | Value |
|-----------|-------|
| Architecture | Swin-T (hybrid CNN + Swin Transformer) |
| Parameters | 27.5M |
| Training data | 15M patches (TCGA + PAIP) |
| Input size | 224x224 |
| License | GPL-3.0 |
| Download size | ~110MB |
| Institutional email? | NO |

### Why CTransPath is great for hackathon
- **Smallest model** (27.5M params) = fastest inference
- **Swin Transformer** = produces multi-scale features = easier for segmentation
- Loads via timm = works with more segmentation frameworks
- GPL-3.0 = freely available, no gating

### Load with timm (requires custom ConvStem)
```python
import timm
import torch.nn as nn
from timm.layers.helpers import to_2tuple

class ConvStem(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768,
                 norm_layer=None, **kwargs):
        super().__init__()
        assert patch_size == 4
        assert embed_dim % 8 == 0
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3,
                                  stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1]
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x

# Load model
model = timm.create_model(
    "hf-hub:1aurent/swin_tiny_patch4_window7_224.CTransPath",
    embed_layer=ConvStem,
    pretrained=True,
)
model = model.eval()

# Extract features
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)
features = model(transforms(image).unsqueeze(0))  # (1, num_features)
```

### Benchmark Results (ROC AUC)
- Camelyon16: 96.3
- TCGA-NSCLC classification: 97.3
- TCGA-RCC classification: 98.9
- TCGA-BRCA histological: 95.8

### CTransPath for SMP segmentation
Because CTransPath is a Swin Transformer, it DOES produce hierarchical multi-scale features. This means it can potentially work with SMP as a custom encoder (see Section 9). However, the custom ConvStem complicates direct SMP integration. **For segmentation, the recommended path is torchseg or a manual UNet decoder.**

---

## 6. Lunit DINO

### URLs
- HuggingFace: https://huggingface.co/1aurent/vit_small_patch8_224.lunit_dino
- HuggingFace collection: https://huggingface.co/collections/1aurent/lunit-models-65639b4149f816b7989185b4
- GitHub: https://github.com/lunit-io/benchmark-ssl-pathology

### Specifications
| Attribute | DINO p16 | DINO p8 |
|-----------|----------|---------|
| Architecture | ViT-S/16 | ViT-S/8 |
| Parameters | ~22M | ~22M |
| Training data | 19M patches (TCGA) | 19M patches (TCGA) |
| Input size | 224x224 | 224x224 |
| License | Lunit non-commercial | Lunit non-commercial |
| Institutional email? | NO | NO |

### Load with timm
```python
import timm

model = timm.create_model(
    "hf-hub:1aurent/vit_small_patch8_224.lunit_dino",
    pretrained=True,
)
model = model.eval()

data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)
output = model(transforms(image).unsqueeze(0))  # (1, num_features)
```

### Verdict
- Smallest model (~22M params) = very fast
- Decent pathology features but outperformed by Phikon-v2 and UNI
- Good fallback if GPU memory is limited
- Patch size 8 version has finer-grained features (more tokens per image)

---

## 7. RetCCL

### URLs
- GitHub: https://github.com/Xiyue-Wang/RetCCL
- Google Drive weights: https://drive.google.com/drive/folders/1AhstAFVqtTqxeS9WlBpU41BV08LYFUnL
- Paper: Medical Image Analysis, 2023

### Specifications
| Attribute | Value |
|-----------|-------|
| Architecture | ResNet-50 (modified) |
| Parameters | ~25M |
| Training data | 15M patches (TCGA + PAIP) |
| License | GPL-3.0 |
| Institutional email? | NO |

### Performance
- TissueNet Acc@1: 67.09% (vs 50.35% ImageNet)
- TCGA-NSCLC MIL accuracy: 0.911
- Colorectal cancer SVM: 98.40%

### Verdict
- ResNet-50 based = works natively with SMP as encoder
- But older model, outperformed by transformer-based models
- **Could be useful if you need a CNN encoder for SMP segmentation**
- Weights need manual download from Google Drive

---

## 8. Complete Comparison Table

| Model | Params | Architecture | Free Download? | Input | Output Dim | Best For |
|-------|--------|-------------|----------------|-------|------------|----------|
| **Phikon-v2** | 300M | ViT-L/16 | YES | 224 | 1024 | Classification (best free) |
| **Phikon** | 86M | ViT-B/16 | YES | 224 | 768 | Classification (lighter) |
| **CTransPath** | 28M | Swin-T | YES | 224 | 768 | Fast classification |
| **Lunit DINO** | 22M | ViT-S/8 | YES | 224 | 384 | Low-resource |
| **RetCCL** | 25M | ResNet-50 | YES | 224 | 2048 | SMP segmentation encoder |
| UNI2-h | 632M | ViT-H/14 | GATED (inst. email) | 224 | 1280 | Best overall |
| Virchow2 | 632M | ViT-H/14 | GATED | 224 | 1280 | Largest dataset |
| CONCH | ~86M | ViT-B + CoCa | GATED (inst. email) | 224 | 512 | Zero-shot |

---

## 9. CRITICAL: Using Pathology Models for Segmentation (SMP / torchseg)

### The Problem
SMP expects encoders that produce multi-scale feature maps (like ResNet: 64ch -> 128ch -> 256ch -> 512ch at decreasing spatial resolutions). ViT models produce a single-scale token sequence. They are NOT directly compatible with SMP.

### Solution A: Use torchseg (fork of SMP with ViT support)
```bash
pip install torchseg
```

```python
import torchseg

# UNet with ViT encoder -- for 128x128 segmentation
model = torchseg.Unet(
    "vit_small_patch16_224",       # or any timm ViT model
    in_channels=3,
    classes=1,                      # binary segmentation
    encoder_depth=4,
    encoder_indices=(2, 5, 8, 11),  # which transformer blocks to tap
    encoder_weights=True,           # ImageNet pretrained
    decoder_channels=(256, 128, 64, 32),
    encoder_params={
        "scale_factors": (4, 2, 1, 0.5),  # spatial rescaling
        "img_size": 128,                    # your input size
    },
)
```

**To use Lunit DINO weights with torchseg:**
```python
model = torchseg.Unet(
    "hf-hub:1aurent/vit_small_patch8_224.lunit_dino",
    in_channels=3,
    classes=1,
    encoder_depth=4,
    encoder_indices=(2, 5, 8, 11),
    encoder_weights=True,
    decoder_channels=(256, 128, 64, 32),
    encoder_params={
        "scale_factors": (4, 2, 1, 0.5),
        "img_size": 128,
    },
)
```

### Solution B: Use CTransPath with SMP (Swin = hierarchical)
CTransPath is a Swin Transformer, which naturally produces multi-scale feature maps. However, it requires the custom ConvStem class. You would need to register it as a custom SMP encoder:

```python
import segmentation_models_pytorch as smp

# Register CTransPath as custom encoder
smp.encoders.encoders["ctranspath"] = {
    "encoder": MyCTransPathEncoder,  # wrapper class you define
    "pretrained_settings": {
        "pathology": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "url": "",
            "input_space": "RGB",
            "input_range": [0, 1],
        },
    },
    "params": {}
}

# Then use normally
model = smp.Unet(encoder_name="ctranspath", classes=1, in_channels=3)
```

This requires writing a wrapper class that:
1. Inherits from `nn.Module` and `smp.encoders.EncoderMixin`
2. Sets `_out_channels`, `_depth`, `_in_channels`
3. Returns features at decreasing spatial resolutions from `forward()`

### Solution C: Use RetCCL (ResNet-50) directly with SMP -- EASIEST for segmentation
```python
import segmentation_models_pytorch as smp
import torch

# Standard SMP with ResNet50
model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
)

# Then load RetCCL weights into the encoder
retccl_weights = torch.load("retccl_best_ckpt.pth", map_location="cpu")
# Filter to only encoder keys and load
encoder_state = {k.replace("module.", ""): v for k, v in retccl_weights.items()
                 if not k.startswith("fc")}
model.encoder.load_state_dict(encoder_state, strict=False)
```

### Solution D: Phikon-v2 features + simple segmentation decoder (RECOMMENDED)
```python
import torch
import torch.nn as nn
from transformers import AutoModel

class PhikonSegmentation(nn.Module):
    """Use Phikon-v2 patch tokens for segmentation."""
    def __init__(self, num_classes=1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("owkin/phikon-v2")

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Phikon-v2 with 224x224 input and patch_size=16 gives 14x14 patch tokens
        # Each token has dim 1024
        self.decoder = nn.Sequential(
            nn.Conv2d(1024, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # 14->28
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # 28->56
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # 56->112
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(size=(128, 128)),  # -> target size
            nn.Conv2d(32, num_classes, 1),
        )

    def forward(self, pixel_values):
        with torch.no_grad():
            outputs = self.backbone(pixel_values=pixel_values)
            # Get all patch tokens (exclude CLS), shape: (B, 196, 1024)
            patch_tokens = outputs.last_hidden_state[:, 1:, :]

        B = patch_tokens.shape[0]
        # Reshape to spatial: (B, 1024, 14, 14)
        features = patch_tokens.transpose(1, 2).reshape(B, 1024, 14, 14)
        return self.decoder(features)
```

---

## 10. Handling 100x100 Images with Foundation Models

All pathology foundation models expect 224x224 input. Your images are 100x100. Options:

### Option 1: Simple Resize (RECOMMENDED for hackathon)
```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # bilinear upscale
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```
This is what the AutoImageProcessor does by default. Works well -- the model was trained to be robust to resolution variations.

### Option 2: Resize with Padding
```python
transform = transforms.Compose([
    transforms.Resize(224),  # scales to 224x224 (from 100x100)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### Option 3: For ViT models, adjust positional embeddings
timm-based models support `img_size` parameter that automatically interpolates positional embeddings:
```python
model = timm.create_model("...", pretrained=True, img_size=100)
```
This avoids upscaling but changes the number of patch tokens. May reduce quality.

### Recommendation
**Use Option 1 (simple resize to 224x224).** The quality loss from upscaling 100->224 is minimal compared to the benefit of using pathology-pretrained features. The AutoImageProcessor handles this automatically.

---

## 11. Feature Extraction + Simple Classifier vs Full Fine-Tuning

### For Hackathon: Feature Extraction + Linear Head (RECOMMENDED)

**Why:**
- 10x faster to train (minutes vs hours)
- Works great with small datasets (<1000 samples per class)
- No risk of overfitting or catastrophic forgetting
- Can use kNN or SVM for even faster iteration

```python
# Step 1: Extract all features once (offline)
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModel
from torch.utils.data import DataLoader

processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
model = AutoModel.from_pretrained("owkin/phikon-v2")
model = model.eval().cuda()

all_features = []
all_labels = []

for images, labels in dataloader:
    inputs = processor(images=[img for img in images], return_tensors="pt").to("cuda")
    with torch.inference_mode():
        features = model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()
    all_features.append(features)
    all_labels.append(labels.numpy())

X = np.concatenate(all_features)
y = np.concatenate(all_labels)

# Step 2: Train simple classifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Option A: Logistic Regression (fast, good)
clf = LogisticRegression(max_iter=1000, C=1.0)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

# Option B: kNN (instant, no training)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
accuracy = knn.score(X_test, y_test)

# Option C: Linear probe in PyTorch (if you need GPU training)
linear_head = nn.Linear(1024, 12).cuda()
optimizer = torch.optim.Adam(linear_head.parameters(), lr=1e-3)
```

### When to Fine-Tune
- Only if linear probe accuracy is insufficient
- Use LoRA fine-tuning, NOT full fine-tuning
- Phikon-v2 has a Colab notebook for LoRA: https://colab.research.google.com/drive/1zjxscEBgpizHBCwMy-aNz2916AVdB642

### Performance Expectation
Based on benchmarks:
- **Linear probe with Phikon-v2**: expect 85-95% accuracy on tissue classification
- **kNN with Phikon-v2**: expect 80-90% accuracy
- **Fine-tuning**: 90-97% accuracy but takes much longer

---

## 12. Hackathon Quick-Start Plan

### Classification Task (100x100, 12 classes)
1. Install: `pip install transformers torch torchvision scikit-learn`
2. Load Phikon-v2 from HuggingFace
3. Extract features for all images (CLS token, dim=1024)
4. Train LogisticRegression or small MLP
5. Expected time: 30 minutes total

### Segmentation Task (128x128, binary)
1. Install: `pip install segmentation-models-pytorch torch`
2. Use SMP with EfficientNet-B0 (ImageNet) as baseline
3. If time permits, try:
   - torchseg with Lunit DINO encoder
   - Custom Phikon decoder (Section 9, Solution D)
4. Expected time: 1-2 hours for baseline

### Priority Order
1. Get Phikon-v2 classification working first (highest impact, fastest)
2. Get SMP segmentation baseline working (EfficientNet encoder)
3. If time permits, try pathology encoder for segmentation
4. If time permits, try LoRA fine-tuning on Phikon-v2

---

## Sources

- Phikon-v2 on HuggingFace: https://huggingface.co/owkin/phikon-v2
- Phikon on HuggingFace: https://huggingface.co/owkin/phikon
- UNI on GitHub: https://github.com/mahmoodlab/UNI
- UNI2-h on HuggingFace: https://huggingface.co/MahmoodLab/UNI2-h
- Virchow2 on HuggingFace: https://huggingface.co/paige-ai/Virchow2
- CONCH on GitHub: https://github.com/mahmoodlab/CONCH
- CTransPath on HuggingFace: https://huggingface.co/1aurent/swin_tiny_patch4_window7_224.CTransPath
- CTransPath on GitHub: https://github.com/Xiyue-Wang/TransPath
- Lunit DINO on HuggingFace: https://huggingface.co/1aurent/vit_small_patch8_224.lunit_dino
- Lunit benchmark GitHub: https://github.com/lunit-io/benchmark-ssl-pathology
- RetCCL on GitHub: https://github.com/Xiyue-Wang/RetCCL
- torchseg library: https://github.com/isaaccorley/torchseg
- SMP custom encoder docs: https://smp.readthedocs.io/en/latest/insights.html
- Pathology foundation models list: https://github.com/georg-wolflein/pathology-foundation-models
- Benchmark paper 2024: https://arxiv.org/html/2410.16038v1
- Clinical benchmark of public pathology FMs: https://pmc.ncbi.nlm.nih.gov/articles/PMC12003829/
