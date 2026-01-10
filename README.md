# 🌼 Flower Image Classification – C1M4 Assignment

## 📌 Overview
This notebook implements a **multi-class image classification** task using **Convolutional Neural Networks (CNNs)** with PyTorch.  
The objective is to train and evaluate deep learning models capable of classifying flower images into their respective categories.

The assignment focuses on:
- Loading and preprocessing image data
- Building CNN-based models
- Training and testing the models
- Visualizing predictions on unseen data

---

## 📁 Notebook File
- **File name**: `(ABDULLAH_ALSALEM)Copy_of_C1M4_Assignment.ipynb`
- **Platform**: Google Colab
- **Framework**: PyTorch

---

## 🗂 Dataset
- **Dataset type**: Flower image dataset
- **Task**: Multi-class classification
- **Input**: RGB images
- **Image size**: `224 × 224`
- **Loading method**: `torchvision.datasets.ImageFolder`

The dataset is structured in a directory-based format where each folder represents a class.

---

## ⚙️ Data Preprocessing
The following preprocessing steps are applied:

- Resize and crop images to a fixed size
- Convert images to PyTorch tensors
- Normalize using ImageNet statistics
- Apply data augmentation during training:
  - Random resized crop
  - Random horizontal flip

These steps improve generalization and model performance.

---

## 🧠 Models Implemented
The notebook explores CNN-based architectures for image classification:

### 🔹 Custom CNN (DeepCNN)
- Built from scratch
- Uses convolutional layers with batch normalization
- Includes max pooling and dropout
- Designed to learn visual features directly from the dataset

### 🔹 MobileNetV2 (Transfer Learning)
- Pretrained on ImageNet
- Feature extractor reused
- Final classifier layer adapted for the dataset
- Provides faster convergence and improved accuracy

---

## 🏋️ Training Configuration
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam
- **Learning Rate**: Configurable (default `1e-4`)
- **Batch Size**: 64
- **Device**: GPU if available, otherwise CPU

The training loop includes:
- Forward pass
- Loss computation
- Backpropagation
- Weight updates

---

## 📊 Evaluation
Model performance is evaluated using:
- Training accuracy and loss
- Test accuracy and loss

Evaluation is performed on unseen data to assess generalization.

---

## 🖼 Visualization of Results
The notebook includes visual inspection of predictions by:
- Displaying test images
- Showing predicted labels vs. true labels

This qualitative evaluation helps verify that the model learns meaningful visual patterns.

---

## 💾 Saving the Model
Trained models can be saved for reuse or inference:

```python
torch.save(model.state_dict(), "model_name.pth")
```

## 🧪 Tools & Libraries
- Python

- PyTorch

- Torchvision

- Google Colab

- Matplotlib

## ✅ Conclusion

This assignment demonstrates the application of CNNs and transfer learning for multi-class image classification.
By comparing a custom CNN with a pretrained MobileNetV2 model, the effectiveness of modern deep learning architectures in visual recognition tasks is highlighted.





# 🌸 Flower Image Classification using CNN Models(C1_M5)

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

