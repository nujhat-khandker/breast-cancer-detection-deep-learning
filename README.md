# Breast Cancer Detection using Deep Learning

A deep learning project that classifies breast ultrasound images into Benign, Malignant, and Normal categories using three CNN architectures — a custom CNN, ResNet50, and MobileNetV2 — with Grad-CAM visualizations for model interpretability.

## Dataset

- Source: BUSI (Breast Ultrasound Images) Dataset, publicly available on Kaggle
- Classes: Benign, Malignant, Normal
- Split:
  - Train: 1103 images
  - Validation: 237 images
  - Test: 238 images

## Models Used

| Model | Description |
|---|---|
| Custom CNN | A simple CNN built from scratch |
| ResNet50 | Pretrained on ImageNet, fine-tuned on BUSI dataset |
| MobileNetV2 | Pretrained on ImageNet, fine-tuned on BUSI dataset |

## Results

| Model | Train Acc (%) | Val Acc (%) | Test Acc (%) | Precision (%) | Recall (%) | F1 Score (%) | RMSE |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| CNN | 69.36 | 62.87 | 60.50 | 70.92 | 60.50 | 61.17 | 0.994 |
| ResNet50 | 87.40 | 85.65 | 83.61 | 84.17 | 83.61 | 83.77 | 0.583 |
| MobileNetV2 | 90.48 | 87.76 | 91.18 | 92.48 | 91.18 | 91.37 | 0.449 |

MobileNetV2 performed the best across all metrics, achieving around 91.18% test accuracy.

## Model Explainability

Grad-CAM was used to visualize which regions of the ultrasound images each model focused on when making predictions, helping check whether the models learn clinically relevant features rather than background noise.

## Tech Stack

- Language: Python
- Frameworks: PyTorch, Torchvision, TensorFlow / Keras
- Evaluation & Data: scikit-learn, Pandas, NumPy
- Visualization: Matplotlib, Seaborn

## Project Structure

```text
breast-cancer-detection-deep-learning/
├── breast_cancer_detection.ipynb   # Main notebook: data loading, training, evaluation, Grad-CAM
└── README.md                       # Project documentation
