# 🌸 Flower Image Classification using CNN Models

## 📌 Project Overview
This project focuses on **multi-class image classification** of flowers using deep learning techniques.  
The goal is to build, train, and evaluate convolutional neural network (CNN) models that can correctly classify flower images into their respective categories.

The project uses the **Oxford Flowers Dataset**, which contains images of flowers belonging to multiple classes, and applies different CNN-based architectures to study performance and behavior.

---

## 🗂 Dataset
- **Dataset**: Oxford Flowers
- **Type**: Multi-class image classification
- **Input**: RGB flower images
- **Classes**: Multiple flower categories
- **Image Size**: Resized to `224 × 224`

The dataset is organized using a directory-based structure compatible with `ImageFolder` in PyTorch.

---

## ⚙️ Data Preprocessing
The following preprocessing steps were applied:

- Resize / crop images to `224 × 224`
- Convert images to tensors
- Normalize images using ImageNet mean and standard deviation
- Apply data augmentation for training:
  - Random resized crop
  - Random horizontal flip

---

## 🧠 Models Used
Multiple CNN architectures were explored:

### 1️⃣ Custom CNN (DeepCNN)
A CNN model built from scratch using:
- Convolutional layers
- Batch Normalization
- ReLU activation
- Max Pooling
- Dropout for regularization
- Fully connected classifier

This model helps in understanding the fundamentals of CNN-based image classification.

### 2️⃣ MobileNetV2 (Transfer Learning)
A lightweight and efficient CNN architecture pretrained on ImageNet:
- Feature extractor reused from pretrained weights
- Final classification layer modified to match the dataset
- Faster training and better generalization compared to training from scratch

---

## 🏋️ Training Setup
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam
- **Learning Rate**: `1e-4` (adjusted when needed)
- **Batch Size**: `64`
- **Device**: GPU (if available) / CPU

Training and evaluation were performed using separate data loaders.

---

## 📊 Evaluation
Model performance was evaluated using:
- Classification accuracy
- Training and test loss
- Visual inspection of predictions

Sample predictions from the test set were visualized to compare:
- **Predicted class**
- **True class**

This helped verify that the model learned meaningful visual features.

---

## 🖼 Visualization
The project includes visual output that displays:
- Test images
- Model predictions
- Ground truth labels

This provides qualitative insight into model performance beyond numerical metrics.

---

## 💾 Model Saving
Trained models can be saved for later use or inference:

```python
torch.save(model.state_dict(), "model_name.pth")
```

