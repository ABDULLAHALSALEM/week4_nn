# 🧩 Deep Learning Foundations – C1M1 Assignment

## 📌 Overview
This notebook represents the **C1M1 assignment**, which focuses on the **foundational concepts of deep learning and PyTorch**.  
The main objective of this module is to understand the basic building blocks required before moving to full model training workflows.

The assignment emphasizes clarity, correctness, and familiarity with the deep learning environment.

---

## 📁 Notebook Information
- **File name**: `C1M1_Assignment.ipynb`
- **Module**: C1M1 – Deep Learning Foundations
- **Framework**: PyTorch
- **Environment**: Google Colab

---

## 🧠 Learning Objectives
This assignment helps in understanding:

- The basic structure of a deep learning project
- How to work with tensors in PyTorch
- The role of datasets and data loaders
- Fundamental neural network components
- Preparing the environment for later deep learning modules

---

## ⚙️ Core Concepts Covered
The notebook introduces and demonstrates:

- Creating and manipulating tensors
- Understanding tensor shapes and data types
- Using PyTorch modules and functions
- Basic operations required for neural networks
- Setting up the workflow for model development

---

## 🔄 Workflow Structure
The notebook follows a simple and clear workflow:

1. Import required libraries
2. Initialize the deep learning environment
3. Work with tensors and basic operations
4. Prepare data-related components
5. Validate outputs and intermediate results

This structure ensures readiness for more advanced modules.

---

## 🧪 Verification
To ensure correctness, the notebook includes:
- Output checks
- Shape and type verification
- Step-by-step execution to validate understanding

---

## 🧰 Tools & Libraries
- Python
- PyTorch
- Google Colab

---

## ✅ Conclusion
This assignment establishes a strong foundation for deep learning using PyTorch.  
By completing this notebook, the essential concepts required for subsequent modules—such as data management, model training, and evaluation—are clearly understood.

---

## 🚀 Next Steps
- Move to data management and preprocessing (C1M3)
- Implement full training workflows (C1M2)
- Explore CNN architectures and transfer learning

---



# 🧠 Deep Learning Workflow – C1M2 Assignment

## 📌 Overview
This notebook implements a complete **deep learning workflow** using **PyTorch** as part of the **C1M2 assignment**.  
The focus of this module is on building, training, and evaluating image classification models, following best practices in model development.

The assignment demonstrates how raw image data is transformed into a trained model through a structured training pipeline.

---

## 📁 Notebook Information
- **File name**: `(ABDULLAH_ALSALEM)C1M2_Assignment (1).ipynb`
- **Module**: C1M2 – Deep Learning Workflow
- **Framework**: PyTorch
- **Environment**: Google Colab

---

## 🗂 Dataset
- **Task**: Multi-class image classification
- **Input**: RGB images
- **Image size**: Standardized before training
- **Loading method**: PyTorch `DataLoader`

The dataset is prepared and loaded in batches to ensure efficient training and evaluation.

---

## ⚙️ Data Preprocessing
The notebook applies essential preprocessing steps, including:

- Image resizing and cropping
- Conversion to tensors
- Normalization
- Data augmentation for training data

These steps help improve model generalization and stability during training.

---

## 🧠 Models Used
The assignment explores CNN-based architectures for image classification, including:

- **Convolutional Neural Networks (CNNs)** built from scratch
- **Pretrained CNN models** using transfer learning (when applicable)

Model architectures are adapted to match the number of target classes.

---

## 🏋️ Training Pipeline
The training workflow includes:

- Forward pass through the model
- Loss computation using `CrossEntropyLoss`
- Backpropagation
- Parameter updates using the Adam optimizer

Key training parameters:
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: Configurable
- **Epochs**: Defined based on experimentation
- **Device**: GPU if available, otherwise CPU

---

## 📊 Evaluation
Model performance is evaluated using:

- Training loss and accuracy
- Test loss and accuracy

Evaluation is performed on unseen data to measure the model’s ability to generalize.

---

## 🖼 Visualization
The notebook includes visualization of:
- Sample images
- Model predictions vs. true labels

This qualitative analysis supports numerical evaluation metrics.

---

## 💾 Model Saving
Trained models can be saved for later use or inference:

```python
torch.save(model.state_dict(), "model_name.pth")
```



# 📊 Image Data Management – C1M3 Assignment

## 📌 Overview
This notebook focuses on **image data management and preparation** as part of the C1M3 assignment.  
The main objective is to correctly load, organize, preprocess, and prepare image datasets for deep learning workflows using **PyTorch**.

The assignment emphasizes data handling rather than model optimization or advanced architectures.

---

## 📁 Notebook Information
- **File name**: `(ABDULLAH_ALSALEM(EDT)Copy_of_C1M3_Assignment.ipynb`
- **Module**: C1M3 – Data Management
- **Framework**: PyTorch
- **Environment**: Google Colab

---

## 🗂 Dataset Handling
- **Task type**: Multi-class image classification
- **Input data**: RGB images
- **Dataset structure**: Directory-based class folders
- **Loading method**: `torchvision.datasets.ImageFolder`

The dataset is organized so that each class is represented by a separate folder, allowing automatic label assignment.

---

## ⚙️ Data Preprocessing
The following preprocessing steps are applied to prepare the images for training:

- Resize images to a fixed resolution
- Convert images to PyTorch tensors
- Normalize images using standard mean and standard deviation values
- Apply data augmentation techniques for training data:
  - Random resized crop
  - Random horizontal flip

These steps ensure consistency and improve generalization in later training stages.

---

## 🔄 Data Loaders
Data loaders are created using `torch.utils.data.DataLoader` to:

- Load data in batches
- Shuffle training data
- Optimize data feeding during training

Key parameters:
- **Batch size**: Configurable
- **Shuffle**: Enabled for training
- **Workers**: Used to speed up data loading

---

## 🧪 Verification and Inspection
To ensure correct data handling, the notebook includes:
- Dataset size checks
- Class count verification
- Visual inspection of sample images

This helps confirm that the dataset is loaded correctly and labels are assigned as expected.

---

## 🎯 Scope of the Assignment
This assignment focuses on:
- Correct dataset structure
- Proper preprocessing
- Efficient data loading

Model training and performance optimization are **not** the primary objectives of this module.

---

## 🧰 Tools & Libraries
- Python
- PyTorch
- Torchvision
- Google Colab
- Matplotlib

---

## ✅ Conclusion
This notebook demonstrates proper image data management practices required for deep learning projects.  
By correctly organizing, preprocessing, and loading the dataset, it establishes a solid foundation for subsequent model training and evaluation stages in later modules.

---





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

