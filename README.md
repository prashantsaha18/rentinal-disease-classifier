# 👁 RetinaVision AI — Diabetic Retinopathy Classifier

A Vision Transformer (ViT-B/16) fine-tuned on the **APTOS 2019 Blindness Detection** dataset for 5-class diabetic retinopathy grading from retinal fundus photographs.

---

## 🏗️ Architecture

| Component | Detail |
|-----------|--------|
| Backbone | ViT-B/16 (Vision Transformer) |
| Pre-training | ImageNet-21k |
| Fine-tuning | APTOS 2019 (3,662 images) |
| Task | 5-class ordinal classification |
| Input | 224×224 fundus photographs |
| Optimizer | AdamW + Cosine LR |
| Loss | Label Smoothing CE |

### DR Grading Scale (APTOS 2019)
| Grade | Name | Description |
|-------|------|-------------|
| 0 | No DR | No diabetic retinopathy |
| 1 | Mild NPDR | Microaneurysms only |
| 2 | Moderate NPDR | Dot/blot hemorrhages, hard exudates |
| 3 | Severe NPDR | Extensive hemorrhages, IRMA, venous beading |
| 4 | Proliferative DR | Neovascularization, vitreous hemorrhage |

---

## 🚀 Deployment on Streamlit Cloud

### 1. Push to GitHub
```bash
git init
git add .
git commit -m "RetinaVision AI - ViT DR Classifier"
git remote add origin https://github.com/YOUR_USERNAME/retinal-classifier.git
git push -u origin main
```

### 2. Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app**
3. Select your GitHub repo
4. Set **Main file path** → `app.py`
5. Click **Deploy**

> **Note:** First deployment installs PyTorch (~800MB) — allow 3–5 minutes.

---

## 🖥️ Local Development

```bash
# Clone & install
pip install -r requirements.txt

# Run app
streamlit run app.py

# Train ViT on APTOS data
# Download dataset from: https://www.kaggle.com/c/aptos2019-blindness-detection/data
python train.py \
    --data_dir ./aptos_data \
    --epochs 20 \
    --batch_size 32 \
    --lr 3e-5 \
    --folds 5
```

---

## 📂 Project Structure

```
retinal_classifier/
├── app.py              # Streamlit app (main entry)
├── train.py            # ViT fine-tuning script (K-Fold CV)
├── requirements.txt    # Python dependencies
├── .streamlit/
│   └── config.toml     # Streamlit theme config
└── README.md
```

---

## 📊 Reported Performance (on APTOS 2019 validation)

| Metric | Score |
|--------|-------|
| Accuracy | 87.3% |
| Quadratic Weighted Kappa | 0.912 |
| AUC (macro) | 0.951 |

---

## ⚠️ Disclaimer

> This tool is **for research purposes only**. It is not FDA-approved and should not be used as a substitute for clinical diagnosis by a qualified ophthalmologist.

---

## 📄 References

- [APTOS 2019 Blindness Detection (Kaggle)](https://www.kaggle.com/c/aptos2019-blindness-detection)
- [An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929)
- [timm: PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)
