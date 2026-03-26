# PRESENTATION PLAYBOOK - WhiteCoat.dev
## AI in Healthcare Hackathon 2026 | March 28, 09:30-14:30

---

## EXACT SCORING (30 points total for Phase 2)

| Category | Points | What Judges Want |
|----------|--------|-----------------|
| User Interface Design | 10 | Clean UI, image input, model output visualization, disclaimer shown |
| Presentation & Clarity | 10 | Clear explanation of approach, methodology, workflow |
| Innovation & Creativity | 5 | Creative AI techniques, efficient pipeline, practical design |
| Error Handling | 5 | Reliability, stability, edge cases handled |

**CRITICAL**: Judges will ask technical questions to verify you UNDERSTAND your own code. If you cannot explain it, you may be disqualified (Section 11 of guidelines).

---

## 1. PRESENTATION STRUCTURE (15 minutes)

### Optimal: 12 slides + live demo

| Section | Time | Slides | Purpose |
|---------|------|--------|---------|
| **Hook + Problem** | 1 min | 1-2 | Grab attention, state why this matters |
| **AI Workflow / Pipeline** | 2.5 min | 3-4 | End-to-end system overview, preprocessing |
| **Model Design** | 3 min | 5-7 | Architecture choices, WHY each model, training strategy |
| **Challenges & Solutions** | 2 min | 8-9 | Class imbalance, what failed, what worked |
| **Results & Visualizations** | 2.5 min | 10-11 | Metrics, confusion matrix, GradCAM, segmentation overlays |
| **Live Demo** | 3 min | 12 (Streamlit) | Upload image, show classification + segmentation |
| **Closing + Disclaimer** | 1 min | 12 | Summary, future work, disclaimer |

**Total: 15 min presentation + Q&A after**

---

## 2. SLIDE-BY-SLIDE SCRIPT

### SLIDE 1: Hook (60 seconds)

**Opening line** (choose one):

> "Imagine you are a pathologist in a rural hospital in Uzbekistan. You have 200 biopsy slides to review today, but you are the only specialist for the entire region. What if AI could help you prioritize the most dangerous cases in seconds?"

OR:

> "Every year, delayed biopsy analysis costs lives -- especially in countries where there is one pathologist for every hundred thousand people. Today we present a system that classifies 12 types of biopsy tissue and segments lesion boundaries in under 2 seconds."

**Content on slide:**
- Team name: WhiteCoat.dev
- One powerful statistic (e.g., pathologist shortage in Central Asia)
- One striking biopsy image with AI overlay
- Title: "AI-Assisted Biopsy Analysis: Classification and Segmentation"

---

### SLIDE 2: Problem Definition (30 seconds)

**What to say:**
- Two tasks: 12-class biopsy classification + binary lesion segmentation
- 11,400 training images for classification, 1,800 for segmentation
- Key challenge: heavily imbalanced classes (331 to 2,136 images per class)
- Goal: accurate, explainable, fast inference

**Visual:** Bar chart of class distribution showing the imbalance

---

### SLIDE 3-4: AI Workflow / Pipeline (2.5 minutes)

**SLIDE 3: Overall Pipeline Diagram**

```
[Biopsy Image] --> [Preprocessing] --> [Classification Branch] --> [12-class prediction + confidence]
                                   --> [Segmentation Branch] --> [Binary mask overlay]
                                   --> [GradCAM Explainability] --> [Attention heatmap]
```

**What to say:**
- "Our system has two parallel branches sharing a common preprocessing pipeline"
- Preprocessing: resize, normalize (ImageNet stats), test-time augmentation
- Classification produces a class label with confidence score
- Segmentation produces a pixel-level binary mask
- GradCAM provides visual explanation of WHERE the model focuses

**SLIDE 4: Data Preprocessing Details**
- Augmentation strategy: RandomRotate90, HorizontalFlip, VerticalFlip, ColorJitter, HueSaturationValue
- "These augmentations are specifically chosen for biopsy images -- rotation invariance matters because tissue orientation on a slide is arbitrary"
- Normalization: ImageNet statistics for transfer learning compatibility
- Image size: matched to model input requirements

---

### SLIDE 5-7: Model Design (3 minutes)

**SLIDE 5: Classification Architecture**

**What to say:**
- "We use EfficientNetV2-S as our primary classifier, with ConvNeXt and Swin Transformer in an ensemble"
- WHY EfficientNetV2: compound scaling (depth, width, resolution) gives best accuracy per FLOP. The V2 variant adds Fused-MBConv blocks for faster training
- WHY ensemble with ConvNeXt + Swin: different architectural biases -- CNN captures local texture, Transformer captures global structure. Biopsy images need BOTH
- Transfer learning from ImageNet: the low-level features (edges, textures, colors) transfer well to medical images

**KEY PHRASE for judges:** "We ensemble these three models because CNNs excel at local texture patterns like cell morphology, while transformers capture global tissue architecture. Biopsy classification needs both scales of information."

**Visual:** Side-by-side architecture diagrams, simple boxes, not complex

**SLIDE 6: Handling Class Imbalance**

This is your **Challenges & Solutions** ammunition:
- Weighted CrossEntropyLoss: weights inversely proportional to class frequency
- Oversampling minority classes (Class 5: 441, Class 8: 331)
- Heavy augmentation on minority classes
- "We tried focal loss but weighted CE gave better results on our validation set"

**Visual:** Before/after bar chart showing balanced effective distribution

**SLIDE 7: Segmentation Architecture**

**What to say:**
- "U-Net++ with EfficientNet-B4 encoder from segmentation_models_pytorch"
- WHY U-Net++: nested dense skip connections capture features at multiple scales better than vanilla U-Net
- WHY EfficientNet-B4 encoder: pretrained backbone gives strong feature extraction even with only 1,800 training images
- Loss: Dice + BCE combination -- Dice handles the foreground/background imbalance, BCE provides stable gradients
- "The combination loss consistently outperforms either loss alone for binary medical segmentation"

**Visual:** U-Net++ architecture diagram with nested skip connections highlighted

---

### SLIDE 8-9: Challenges & Solutions (2 minutes)

**SLIDE 8: Technical Challenges**

| Challenge | Solution | Result |
|-----------|----------|--------|
| Severe class imbalance (6.5x ratio) | Weighted loss + oversampling + augmentation | Balanced per-class recall |
| Small segmentation dataset (1,800) | Transfer learning + heavy augmentation + U-Net++ | High IoU on validation |
| Anonymous classes (no disease names) | Treated as purely visual classification, used confusion matrix to find similar classes | Identified merge candidates |
| Overfitting risk | Early stopping, dropout, weight decay, 5-fold CV | Gap < 3% train/val |

**SLIDE 9: What We Tried That Failed (IMPORTANT -- shows honesty)**

- "We initially tried a single EfficientNet-B4 and achieved X% accuracy. Ensembling pushed it to Y%"
- "We tried larger input resolutions but ran into VRAM limits on our hardware"
- "Focal loss underperformed weighted CE for our specific class distribution"
- "We experimented with stain normalization but the biopsy images did not show strong color variation across classes"

**WHY THIS MATTERS:** Judges respect teams that show scientific rigor. Showing failed experiments proves you understand the problem deeply, not just copied code.

---

### SLIDE 10-11: Results & Visualizations (2.5 minutes)

**SLIDE 10: Quantitative Results**

**Must show:**
- Overall accuracy (classification) and Mean IoU (segmentation)
- Per-class accuracy or F1 as a bar chart
- Confusion matrix (12x12 heatmap) -- point out which classes are most confused
- Training curves (loss and accuracy over epochs) showing convergence, no overfitting

**What to say:**
- "Our classification model achieves X% accuracy on the validation set"
- "Segmentation achieves Y Mean IoU on the 400 validation images"
- "The confusion matrix reveals that classes 3 and 6 are most often confused, which suggests they share similar tissue morphology"

**SLIDE 11: Qualitative Results (VISUAL IMPACT)**

This slide wins or loses the presentation. Show:

1. **GradCAM heatmaps** (3-4 examples): Original biopsy | GradCAM overlay | Predicted class
   - Show the model focusing on the CORRECT region (cell clusters, tissue boundaries)
   - "The model attends to the cellular region, not background artifacts"

2. **Segmentation overlays** (3-4 examples): Original | Ground truth mask | Predicted mask | Overlay
   - Choose examples where prediction closely matches ground truth
   - Include one challenging case where the model still performs well

3. **One failure case**: Show an image the model gets wrong, explain WHY
   - "This image from class 8 was misclassified as class 3. The tissue structure is visually ambiguous even to trained pathologists"

---

### SLIDE 12: Live Demo + Closing (3 + 1 minutes)

**Switch to Streamlit. Demo script below in Section 6.**

After demo, return to final slide:
- Summary: "Two-task system for biopsy analysis: 12-class classification with ensemble and binary segmentation with U-Net++"
- Future work: "Uncertainty quantification, larger datasets, clinical validation"
- **MANDATORY disclaimer**: "For research and demonstration purposes only. Not for clinical use."
- "Thank you. We are WhiteCoat.dev. Questions?"

---

## 3. VISUALIZATIONS THAT IMPRESS JUDGES

### Priority order (must-have first):

1. **GradCAM heatmaps** -- THE most impressive single visualization. Shows the model is not a black box. Generate with:
   ```python
   # Use pytorch-grad-cam library
   from pytorch_grad_cam import GradCAM
   from pytorch_grad_cam.utils.image import show_cam_on_image

   cam = GradCAM(model=model, target_layers=[model.features[-1]])
   grayscale_cam = cam(input_tensor=input_tensor)
   visualization = show_cam_on_image(rgb_img, grayscale_cam[0], use_rgb=True)
   ```

2. **Confusion matrix heatmap** -- Use seaborn with annotations. Make it large, readable. Color-code by accuracy.

3. **Segmentation overlay comparison** -- Side-by-side: input | ground truth | prediction | overlay with green=correct, red=errors

4. **Class distribution bar chart** -- Shows you understood the imbalance challenge

5. **Training curves** -- Loss and metric over epochs. Show no overfitting (train/val gap).

6. **Per-class F1 bar chart** -- Shows which classes are hardest

### Nice-to-have (Innovation bonus):

7. **t-SNE / UMAP of feature embeddings** -- Extract penultimate layer features, color by class. If clusters are clean, it shows the model learned meaningful representations.
   ```python
   from sklearn.manifold import TSNE
   # Extract features from penultimate layer for all val images
   # Plot with class-colored dots
   ```

8. **Confidence calibration plot** -- Expected accuracy vs predicted confidence. Shows if model confidence is trustworthy.

9. **MC Dropout uncertainty** -- Run inference 20 times with dropout enabled, show variance as uncertainty. Highlight: high uncertainty correlates with misclassifications.

---

## 4. INNOVATION IDEAS (5 points for Innovation & Creativity)

### Tier 1: Implement these (high impact, feasible in hours)

1. **GradCAM Explainability** -- Already mentioned. This alone can score Innovation points because it addresses the "black box" criticism of medical AI.

2. **Uncertainty Quantification via MC Dropout** -- Run inference N times with dropout enabled. Report mean prediction and standard deviation. Flag images where uncertainty > threshold for human review. Script:
   ```python
   model.train()  # enables dropout
   predictions = []
   for _ in range(20):
       with torch.no_grad():
           pred = model(image)
           predictions.append(pred.softmax(dim=1))
   mean_pred = torch.stack(predictions).mean(dim=0)
   uncertainty = torch.stack(predictions).std(dim=0).mean()
   ```
   **Why judges love this:** It shows you understand that AI should know when it does NOT know.

3. **Error Handling in UI** -- Validate uploaded image format, size, type. Show meaningful error messages. Handle model loading failures gracefully. This directly scores 5 points.

### Tier 2: Mention in presentation (shows awareness)

4. **Fairness considerations** -- "In a full clinical pipeline, we would evaluate performance across different staining protocols and tissue preparation methods to ensure equitable performance."

5. **Similar case retrieval** -- "Our feature embeddings could power a retrieval system: given a new biopsy, find the 5 most similar cases from the training set as reference."

6. **Mobile/edge deployment** -- "EfficientNetV2-S was chosen partly for its efficiency. The model could be quantized to INT8 for deployment on edge devices in resource-limited settings."

---

## 5. Q&A PREPARATION: 20 Most Likely Questions

### Technical Questions (most likely from judges)

**Q1: Why did you choose EfficientNetV2-S over other architectures?**
> "EfficientNetV2 uses compound scaling to optimize depth, width, and resolution together. The V2 variant introduced Fused-MBConv blocks which are faster on modern hardware. The S variant balances accuracy and computational cost -- important for a practical medical tool. It also has strong ImageNet pretrained weights which transfer well to medical imaging."

**Q2: How do you handle class imbalance?**
> "Three strategies working together: weighted CrossEntropy loss with weights inversely proportional to class frequency, oversampling of minority classes during data loading, and heavier augmentation on minority classes. We validated that this combination brings per-class recall within a reasonable range. We also tried focal loss but weighted CE performed better on our validation set."

**Q3: Why ensemble three models? Is it not overkill?**
> "Each architecture has different inductive biases. EfficientNet captures local texture efficiently, ConvNeXt brings modern CNN design with larger kernels, and Swin Transformer captures long-range dependencies through shifted windows. For biopsy images where both cellular detail and tissue architecture matter, this diversity improves predictions. The ensemble improves accuracy by X% over the best single model."

**Q4: How does U-Net++ differ from standard U-Net?**
> "U-Net++ adds nested dense skip connections between the encoder and decoder. Instead of direct skip connections, features pass through a series of dense convolution blocks at different semantic levels. This means the decoder receives features at multiple scales, which is critical for segmentation where lesion boundaries can be both fine-grained and large-scale."

**Q5: What loss function did you use for segmentation and why?**
> "A combination of Dice loss and Binary Cross Entropy. Dice loss directly optimizes the IoU-like metric, handling foreground/background imbalance naturally. BCE provides stable pixel-level gradients. The combination consistently outperforms either alone in binary medical segmentation tasks."

**Q6: How do you prevent overfitting with only 1,800 segmentation images?**
> "Four strategies: transfer learning from ImageNet-pretrained encoder, heavy augmentation specific to biopsy images, early stopping based on validation IoU, and weight decay regularization. The gap between training and validation metrics is under 3 percentage points."

**Q7: Explain your preprocessing pipeline.**
> "Images are resized to match model input requirements, then augmented with rotation, flipping, color jitter, and hue-saturation adjustments. These are specifically chosen for biopsy images because tissue orientation on a slide is arbitrary, making rotation invariance critical. We normalize using ImageNet statistics because our encoders are pretrained on ImageNet."

**Q8: What is GradCAM and why is it important?**
> "Gradient-weighted Class Activation Mapping. It uses the gradients flowing into the final convolutional layer to produce a heatmap highlighting regions the model considers important for its prediction. For medical AI, this is essential: a doctor needs to verify the model is looking at the correct tissue region, not background artifacts. It builds trust and supports clinical decision-making."

**Q9: What is your test-time augmentation strategy?**
> "We apply 4-8 augmentations at inference: original, horizontal flip, vertical flip, and 90/180/270 degree rotations. Predictions are averaged. This gives a free accuracy boost of typically 0.5-1.5% because it reduces sensitivity to image orientation."

**Q10: How does your inference script work?**
> "The script loads the saved model checkpoint, accepts a directory of test images, applies the same preprocessing as training, runs inference with TTA, and outputs the predictions in the required format. It performs inference only -- no training or weight updates happen."

### Clinical / Applied Questions

**Q11: Can this system replace a pathologist?**
> "Absolutely not, and that is not the goal. This is a decision-support tool. It can help prioritize cases, flag suspicious slides, and provide a second opinion. The final diagnosis must always be made by a qualified pathologist. That is why our interface includes the disclaimer: 'For research and demonstration purposes only.'"

**Q12: What about patient data privacy?**
> "The system processes anonymized biopsy images with no patient metadata. In a clinical deployment, we would need HIPAA/GDPR compliance, data encryption, and audit trails. The model itself stores no patient information."

**Q13: How would you validate this for clinical use?**
> "Multi-center prospective clinical trials with diverse patient populations, comparison against board-certified pathologists, regulatory approval through relevant medical device frameworks, and continuous performance monitoring after deployment."

**Q14: The classes are anonymous. How do you handle that?**
> "We treated this as a purely visual pattern recognition task. We used the confusion matrix to understand which classes the model finds similar. Interestingly, classes that are most confused likely represent conditions with overlapping histological features. In clinical deployment, knowing the actual disease names would let us incorporate domain-specific knowledge."

### System / Engineering Questions

**Q15: Why Streamlit for the interface?**
> "Rapid prototyping. Streamlit lets us build an interactive web application in pure Python, which matches our team's expertise. It supports image upload, real-time visualization, and runs locally without complex deployment. For a hackathon demo, speed of development is critical."

**Q16: How fast is inference?**
> "Single image inference takes approximately X seconds on [your hardware]. The bottleneck is the ensemble averaging across three models. For real-time clinical use, we could use model distillation to compress the ensemble into a single model, or quantize to INT8."

**Q17: What hardware did you train on?**
> "[State your actual hardware]. Training the classification ensemble took approximately X hours. The segmentation model trained in Y hours. We used mixed-precision training (FP16) to maximize GPU utilization."

**Q18: What framework and libraries did you use?**
> "PyTorch for deep learning, timm for model architectures, segmentation_models_pytorch for U-Net++, albumentations for augmentation, pytorch-grad-cam for explainability, and Streamlit for the interface. All open-source, all reproducible."

### Tough / Adversarial Questions

**Q19: What is the weakest part of your solution?**
> "Honestly, the class imbalance. Classes 5 and 8 have very few samples (441 and 331). Despite our mitigation strategies, per-class accuracy for minority classes remains lower. With more data or synthetic augmentation via diffusion models, this could be improved significantly."

**Q20: How do you know the model is not just memorizing the training data?**
> "Three evidences: the train/val accuracy gap is less than 3%, the confusion patterns make clinical sense -- similar tissue types are confused, not random classes -- and GradCAM shows the model focuses on biologically meaningful regions rather than image artifacts. We also use 5-fold cross-validation to ensure robustness."

---

## 6. LIVE DEMO SCRIPT (3 minutes)

### Pre-Demo Checklist (do 30 minutes before presentation)

- [ ] Streamlit app starts without errors
- [ ] All model checkpoints loaded successfully
- [ ] Test with 5 different images -- all return results
- [ ] Browser zoomed to 125% for projector visibility
- [ ] Disable all notifications on your laptop
- [ ] Close all unnecessary applications
- [ ] Have backup screenshots ready in a slide
- [ ] Pre-load the Streamlit tab so there is no startup delay

### Demo Script (3 minutes)

**Minute 1: Classification Demo**

1. "Let me show you the system in action."
2. Open Streamlit (should be pre-loaded in browser tab).
3. Point out the disclaimer: "For research and demonstration purposes only."
4. Upload a biopsy image from CLASS 7 (largest class, most reliable prediction).
5. "The model classifies this as Class 7 with 94% confidence."
6. Show the GradCAM heatmap: "Notice how the model focuses on the cellular region, not the background."
7. Show the confidence bar chart for all 12 classes.

**Minute 2: Segmentation Demo**

1. Switch to segmentation tab.
2. Upload a segmentation test image.
3. "Here we see the original biopsy image, and the predicted lesion boundary."
4. Show the overlay: "The green overlay indicates the predicted lesion region."
5. "This runs in under X seconds."

**Minute 3: Edge Cases + Error Handling**

1. Upload a deliberately wrong image (a photo of text or a non-medical image).
2. "Notice the system handles invalid inputs gracefully -- it shows an error message rather than crashing."
3. Upload a challenging biopsy image (minority class).
4. "For this more difficult case, the model shows lower confidence -- 67%. In a clinical setting, this would be flagged for expert review."
5. Show uncertainty visualization if implemented.

### IF THE DEMO CRASHES

**Plan A: Smooth pivot (2 seconds)**
> "Let me switch to our pre-recorded demonstration while I address this."
> Switch to backup slide with screenshots. Continue narrating as if nothing happened.

**Plan B: Quick restart (10 seconds)**
> "One moment -- let me restart the application."
> Have a second terminal tab with `streamlit run app.py` ready to paste.

**Plan C: Screenshot walkthrough (immediate)**
> "The live environment is having a technical issue, but let me walk you through the exact same workflow using these screenshots."
> Use 4-5 prepared screenshots showing the full flow.

**NEVER:**
- Spend more than 15 seconds trying to fix a live issue
- Apologize excessively
- Say "this was working 5 minutes ago"
- Debug on screen

---

## 7. PRESENTATION DELIVERY TIPS

### Speaking Style

- **Pace**: Slightly slower than conversational. Judges are evaluating technical content.
- **Jargon balance**: Use technical terms but immediately explain them. "We use GradCAM -- Gradient-weighted Class Activation Mapping -- which shows WHERE the model looks when making its decision."
- **Eye contact**: Look at judges, not at slides. The slides are your visual aid, not your script.
- **Pointer**: Use a laser pointer or cursor to highlight specific areas of visualizations.

### What Makes You Stand Out as a Medical Student

Leverage your clinical background:
- "As a medical student, I understand that a pathologist's workflow requires tools that explain their reasoning, not just give answers."
- "From our clinical training, we know that biopsy analysis depends heavily on tissue architecture and cellular morphology -- our model captures both through the CNN-Transformer ensemble."
- "In Uzbekistan and Central Asia, the ratio of pathologists to patients is critically low. A tool like this could serve as a first-pass screening system."

### Body Language

- Stand to the side of the screen, not in front of it
- Use your hand to point at specific parts of diagrams
- During the demo, narrate what you are doing: "Now I am uploading a biopsy image..."
- Face the judges when explaining, face the screen only when pointing

### Time Management

- Practice with a timer. If you hit 12 minutes and have not started the demo, skip the "What Failed" slide and go straight to demo.
- Have a teammate signal at 10 minutes, 13 minutes, and 14 minutes.
- The demo is worth 10 points (UI). Do NOT sacrifice demo time for extra slides.

---

## 8. SLIDE DESIGN PRINCIPLES

### Color Scheme
- Dark background (navy or dark gray) with white text for projector visibility
- Accent color: medical blue or teal
- Red ONLY for errors / failure cases
- Green ONLY for correct predictions / success

### Typography
- Title: 36-40pt bold
- Body: 24-28pt
- Code/technical: 20pt monospace
- NEVER go below 20pt -- it will be unreadable on a projector

### Content Density
- Maximum 6 bullet points per slide
- Maximum 30 words per slide (excluding diagrams)
- One idea per slide
- Let visuals do the heavy lifting

### Must Include on Every Slide
- Team name in header or footer: WhiteCoat.dev
- Slide number

### Final Slide Must Include
- "For research and demonstration purposes only. Not for clinical use." (MANDATORY per guidelines)
- Team name and contact
- "Thank you. Questions?"

---

## 9. NIGHT-BEFORE CHECKLIST

### Technical
- [ ] All models trained and saved
- [ ] Inference scripts tested on fresh images
- [ ] Streamlit app runs end-to-end without errors
- [ ] GradCAM visualization generates correctly
- [ ] Confusion matrix and all plots saved as images
- [ ] Backup screenshots captured for every demo step
- [ ] Demo video recorded (2 min) as ultimate backup

### Presentation
- [ ] All slides complete, spell-checked
- [ ] Slide order matches the 15-minute time allocation
- [ ] Disclaimer appears on interface AND final slide
- [ ] Presenter has rehearsed at least 3 times with timer
- [ ] Q&A answers reviewed by all team members
- [ ] Team member roles assigned (presenter, demo operator, Q&A backup)

### Hardware
- [ ] Laptop charged and charger packed
- [ ] HDMI/USB-C adapter for projector tested
- [ ] Internet connection NOT required for demo (everything runs locally)
- [ ] Screen resolution set for external display
- [ ] Notifications disabled
- [ ] Backup of all files on USB drive

---

## 10. WINNING MINDSET

### What separates 1st place from 5th place in hackathon presentations:

1. **Storytelling**: Do not just list features. Tell the story of your journey: problem, approach, failures, improvements, results.

2. **Honesty about limitations**: Judges can spot polished lies. Saying "class 8 accuracy is our weakest point at X%" is more impressive than pretending everything is perfect.

3. **Clinical relevance**: You have a medical student on the team. Use that advantage. Other CS-only teams cannot speak to the clinical workflow.

4. **Smooth demo**: A working demo with clean UI beats a technically superior solution with a broken demo. Every time.

5. **Composure under questions**: If you do not know an answer, say "That is an excellent question. We did not explore that in this hackathon, but our approach would be..." Never bluff technical details.

---

*Prepared for WhiteCoat.dev | AI in Healthcare Hackathon 2026*
*Presentation date: March 28, 2026 | 09:30-14:30*
