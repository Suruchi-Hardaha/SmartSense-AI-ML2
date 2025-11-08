# 🏠 SmartSense Phase 1: Floorplan Parsing with Object Detection

This phase focuses on training a computer-vision model to **parse floorplan images** and extract structured attributes such as:
- Number of rooms, halls, kitchens, bathrooms
- Optional per-room details (labels, areas)

---

## 📂 Project Structure

self_working/
└── notebooks/
├── task1/
│ ├── phase1_floorplan_model_pytorch.ipynb # Training & evaluation notebook
│ ├── floorplan_model_weights.pth # Trained model weights
│ ├── parse_floorplan.py # Inference script → JSON output
│ └── results/ # Evaluation results & visualizations
└── train-val_dataset_final.coco/
├── train/ # Annotated training dataset (COCO format)
└── valid/ # Annotated validation dataset (COCO format)



## ⚙️ Model Details

**Architecture:** Faster R-CNN with ResNet-50-FPN backbone  
**Framework:** PyTorch (torchvision.models.detection)  
**Classes:**
1: bathroom
2: bedroom
3: garage
4: hall
5: kitchen
6: laundry
7: porch
8: room



## 🧠 Training Configuration

| Parameter        | Value |
|------------------|--------|
| Epochs           | 50 |
| Batch size       | 4 |
| Learning rate    | 0.005 |
| Weight decay     | 0.0005 |
| Optimizer        | SGD |
| Loss             | Classification + Regression (per epoch printed) |
## 📦 Model Weights
The trained model weights (~158 MB) can be downloaded from Google Drive:

👉 [Download floorplan_model_weights.pth](https://drive.google.com/file/d/1_hluPXwpSVp6NNV97L8QagRn3SzhAaR4/view?usp=sharing)

**During training:**
- Each epoch prints total classification and regression loss.
- Validation loss monitored for overfitting.
- Best model checkpoint saved automatically.

---

## 🧩 Dataset Split

Data was manually annotated in COCO format and split into:
- **Train:** 60%
- **Validation:** 20%
- **Test:** 20%

| Split | Path | Description |
|-------|------|--------------|
| Train | `notebooks/train-val_dataset_final.coco/train` | Annotated floorplan images |
| Val   | `notebooks/train-val_dataset_final.coco/valid` | Annotated validation images |

---

## 📊 Evaluation Metrics (Validation Set)

| Metric | Description | Value |
|---------|--------------|--------|
| **Mean IoU** | Intersection-over-Union between predicted & true boxes | **0.496** |
| **Count Accuracy** | Per-class correctness of predicted object counts | See below |

### Per-Class Count Accuracy (IoU threshold = 0.6)

| Class | GT Count | Pred Count | Correct | Accuracy |
|:------|----------:|------------:|---------:|----------:|
| bathroom | 98 | 298 | 86 | 0.88 |
| bedroom | 196 | 270 | 192 | 0.98 |
| garage | 75 | 160 | 100 | 1.33 |
| hall | 108 | 296 | 156 | 1.44 |
| kitchen | 90 | 212 | 103 | 1.14 |
| laundry | 32 | 140 | 51 | 1.59 |
| porch | 108 | 305 | 119 | 1.10 |
| room | 35 | 342 | 185 | 5.29 |

---

## 🧾 Inference Script

Run inference on a single floorplan image and get a **structured JSON output**: