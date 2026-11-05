# CNN – CIFAR-10 Image Classification Using Deep Learning

## Project Overview

This project focuses on implementing and understanding Convolutional Neural Networks (CNNs) for image classification using the CIFAR-10 dataset.

It explores fundamental CNN concepts, including convolution operations, padding, strides, pooling, feature extraction, flattening, and backpropagation. The project also implements LeNet-5 and a custom CNN architecture from scratch, followed by experiments involving data augmentation and transfer learning using MobileNetV2.

Multiple model configurations are trained and evaluated to understand how different architectures and training strategies influence image classification performance.

The primary objective is to develop a practical understanding of CNN architecture design, image feature extraction, model optimization, and transfer learning through a multi-class image classification problem.

## Project Objectives

- Understand the fundamentals of Convolutional Neural Networks.
- Load and preprocess the CIFAR-10 dataset.
- Perform exploratory data analysis and visualize image distributions.
- Understand convolution operations, padding, and strides.
- Explore MaxPooling and AveragePooling.
- Implement and analyze LeNet-5 architecture.
- Build a custom CNN architecture from scratch.
- Understand forward propagation and backpropagation in CNNs.
- Train CNN models using EarlyStopping and ReduceLROnPlateau.
- Evaluate models using classification metrics and confusion matrices.
- Implement data augmentation using ImageDataGenerator.
- Explore Keras preprocessing and augmentation layers.
- Apply transfer learning using pretrained MobileNetV2.
- Compare feature extraction and fine-tuning approaches.
- Implement a basic residual block.
- Compare six trained model configurations based on test accuracy, parameter count, and training cost.

## Dataset – CIFAR-10

**Dataset:** CIFAR-10

The CIFAR-10 dataset is a widely used image classification dataset containing 60,000 colour images distributed across 10 different classes.

Each image has a resolution of 32 × 32 pixels with three colour channels (RGB).

### Dataset Distribution

| Dataset | Number of Images |
|---|---:|
| Training Set | 50,000 |
| Testing Set | 10,000 |
| Total | 60,000 |

### CIFAR-10 Classes

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

### Dataset Preprocessing

- Load the CIFAR-10 dataset using TensorFlow/Keras.
- Inspect training and testing dataset shapes.
- Visualize image samples from all 10 classes.
- Analyze pixel intensity distributions using histograms.
- Normalize image pixel values to the range [0, 1].
- Convert integer class labels into one-hot encoded vectors.
- Prepare training and testing datasets for CNN models.

## Technologies & Libraries

### Programming Language
- Python

### Deep Learning Framework
- TensorFlow
- Keras

### Data Processing
- NumPy
- Pandas
- SciPy

### Data Visualization
- Matplotlib
- Seaborn

### Machine Learning Evaluation
- Scikit-learn

### Development Environment
- Jupyter Notebook

## Project Workflow

### Task A1 – Data Loading, EDA & Preprocessing

This section focuses on understanding the CIFAR-10 dataset and preparing images for CNN training.

**1. Dataset Loading**

- Load CIFAR-10 using `keras.datasets.cifar10.load_data()`.
- Inspect training and testing dimensions.
- Examine image data types and class distributions.
- Identify the 10 classification categories.

**2. Exploratory Data Analysis**

- Display a 5 × 10 image grid containing samples from all 10 classes.
- Analyze image dimensions and colour channels.
- Visualize pixel intensity distributions before normalization.
- Examine similarities between visually related classes.

**3. Data Preprocessing**

- Normalize pixel values by dividing by 255.
- Convert categorical labels into one-hot encoded vectors.
- Prepare training and testing data for neural network training.

### Task A2 – Convolution Operation, Padding & Strides

This section explores how convolution extracts meaningful features from images.

**1. Manual Convolution**

- Apply a manually defined edge-detection kernel.
- Perform convolution using SciPy.
- Generate feature maps using the convolution operation.
- Understand how filters identify edges and image patterns.

**2. Padding Techniques**

Compare two major padding strategies:

- **VALID Padding:** Performs convolution without adding additional border pixels, reducing spatial dimensions.
- **SAME Padding:** Adds padding around the image to preserve spatial dimensions when stride is 1.

**3. Stride Comparison**

Experiment with:

- Stride = 1
- Stride = 2

Understand how stride affects feature-map dimensions and computational requirements.

**4. Feature Map Visualization**

- Build a Conv2D layer with 16 filters.
- Generate 16 feature maps.
- Visualize the outputs in a 4 × 4 grid.
- Understand how convolution filters respond to different image patterns.

### Task A3 – Pooling, Flatten & CNN Architectures

This section explores dimensionality reduction and CNN architecture development.

**1. Pooling Operations**

Compare:

- MaxPooling2D
- AveragePooling2D
- GlobalAveragePooling2D

Understand how pooling reduces spatial dimensions while retaining important information.

**2. Flatten Layer**

- Convert multidimensional feature maps into a one-dimensional vector.
- Examine tensor dimensions before and after flattening.
- Understand the relationship between Flatten and Dense layers.

**3. LeNet-5 Architecture**

Implement a LeNet-5-inspired CNN architecture adapted for CIFAR-10.

The architecture includes:

- Convolutional layers.
- Tanh activation functions.
- Average pooling layers.
- Flatten layer.
- Fully connected Dense layers.
- Softmax output layer for 10-class classification.

**4. Custom CNN Architecture**

Build a modern custom CNN using:

- Multiple Conv2D layers.
- ReLU activation.
- MaxPooling2D.
- Increasing filter sizes: 32, 64, and 128.
- GlobalAveragePooling2D.
- Dense layer with 256 neurons.
- Softmax output layer.

The custom architecture uses GlobalAveragePooling2D to keep the parameter count manageable.

### Task A4 – Backpropagation in CNN & Model Training

This section explores how CNNs learn through forward and backward propagation.

**1. Forward Propagation**

Understand the flow of image information through:

- Convolution.
- Activation functions.
- Pooling.
- Feature extraction.
- Flattening or Global Average Pooling.
- Fully connected layers.
- Softmax output.
- Loss calculation.

**2. Backpropagation**

Explore how gradients propagate through:

- Dense layers.
- Convolutional layers.
- MaxPooling layers.
- AveragePooling layers.
- Flatten layers.

Understand how gradient calculations update trainable parameters to minimize the loss function.

**3. Model Compilation and Training**

Compile the custom CNN using:

- Optimizer: Adam.
- Loss Function: Categorical Cross-Entropy.
- Evaluation Metric: Accuracy.
- Batch Size: 64.
- Maximum Epochs: 100.

**4. Training Callbacks**

Implement:

- **EarlyStopping:** Stops training when validation loss stops improving and restores the best weights.
- **ReduceLROnPlateau:** Reduces the learning rate when validation loss reaches a plateau.

**5. Model Evaluation**

Evaluate the trained CNN using:

- Test Loss.
- Test Accuracy.
- Classification Report.
- Confusion Matrix.
- Training and Validation Accuracy Curves.
- Training and Validation Loss Curves.

**6. Error Analysis**

- Display 10 misclassified test images.
- Compare actual and predicted class labels.
- Identify frequently confused class pairs.
- Analyze classification difficulties between visually similar objects.

### Task A5 – Data Augmentation

This section explores how image transformations can improve model generalization.

**1. ImageDataGenerator**

Implement image augmentation using:

- Rotation Range: 15 degrees.
- Width Shift: 0.1.
- Height Shift: 0.1.
- Horizontal Flip.
- Zoom Range: 0.1.
- Fill Mode: Nearest.

Generate nine randomly transformed versions of an original image and visualize them in a 2 × 5 grid.

**2. CNN Training with ImageDataGenerator**

- Train the same custom CNN architecture using augmented images.
- Apply EarlyStopping and ReduceLROnPlateau.
- Evaluate the augmented model on the test dataset.

**3. Keras Data Augmentation Layers**

Implement built-in Keras augmentation layers:

- RandomFlip.
- RandomRotation.
- RandomZoom.
- RandomTranslation.

Integrate these layers directly into the custom CNN architecture.

**4. Performance Comparison**

Compare:

- CNN without augmentation.
- CNN with ImageDataGenerator.
- CNN with Keras augmentation layers.

Analyze training loss, validation loss, accuracy, and test performance.

### Task A6 – Transfer Learning: Feature Extraction & Fine-Tuning

This section explores transfer learning using the pretrained MobileNetV2 architecture.

**1. MobileNetV2 Model Initialization**

- Load pretrained MobileNetV2 using ImageNet weights.
- Remove the original classification head using `include_top=False`.
- Resize CIFAR-10 images to 96 × 96.
- Apply the required MobileNetV2 input preprocessing.
- Freeze the pretrained backbone initially.

**2. Feature Extraction**

Use MobileNetV2 as a fixed feature extractor.

Architecture:

- Pretrained MobileNetV2 backbone.
- GlobalAveragePooling2D.
- Dense layer with 128 neurons.
- Softmax output layer with 10 classes.

Train the newly added classification layers while keeping the pretrained backbone frozen.

**3. Fine-Tuning**

- Unfreeze the pretrained MobileNetV2 backbone.
- Make the top 30 layers trainable.
- Keep BatchNormalization layers frozen.
- Reduce the learning rate to 0.00001.
- Fine-tune the network using the CIFAR-10 dataset.

**4. Feature Extraction vs Fine-Tuning**

Compare:

- Trainable parameters.
- Non-trainable parameters.
- Training accuracy.
- Validation accuracy.
- Training loss.
- Validation loss.
- Test accuracy.

**5. Residual Block Implementation**

Implement a basic residual block using the Keras Functional API.

The architecture includes:

- A 1 × 1 convolutional shortcut projection.
- Two 3 × 3 convolutional layers.
- Add operation to combine the shortcut and main paths.
- ReLU activation.

Understand how residual connections improve gradient flow and support the training of deeper neural networks.

## Model Comparison

The project evaluates six model configurations:

| Model | Approach |
|---|---|
| LeNet-5 | Traditional CNN |
| CNN Scratch No Augmentation | Custom CNN |
| CNN + ImageDataGenerator | CNN with augmentation |
| CNN + Keras Augmentation Layers | CNN with built-in augmentation |
| MobileNetV2 Feature Extraction | Transfer learning |
| MobileNetV2 Fine-Tuned | Transfer learning with fine-tuning |

Models are compared using:

- Test Accuracy.
- Number of Parameters.
- Training Time.
- Training and Validation Performance.

The best-performing model should be selected based on actual measured results rather than assuming that a more complex architecture always performs better.

## Evaluation Metrics

The following evaluation techniques are used:

- **Accuracy:** Measures the percentage of correctly classified images.
- **Loss:** Measures the difference between predicted probabilities and actual labels.
- **Confusion Matrix:** Shows correct and incorrect predictions across all 10 classes.
- **Precision:** Measures the proportion of correct positive predictions.
- **Recall:** Measures the proportion of actual class instances correctly identified.
- **F1-Score:** Provides a balance between Precision and Recall.

Additional visualizations include:

- Pixel intensity histograms.
- Training and validation loss curves.
- Training and validation accuracy curves.
- Confusion matrix heatmaps.
- Misclassified image grids.
- Model comparison tables.

## Project Structure

```text
CNN-CIFAR10-Image-Classification/
│
├── DL_PR4A.ipynb
│
├── README.md
│
├── requirements.txt
│
├── outputs/
│   ├── dataset_visualization.png
│   ├── pixel_histograms.png
│   ├── convolution_feature_maps.png
│   ├── pooling_comparison.png
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── misclassified_images.png
│   ├── augmentation_comparison.png
│   └── model_comparison.png
│
└── models/
    └── trained_model_files
```

**Note:** The `outputs/` and `models/` directories are suggested organizational structures. Create them when exporting visualizations or saving trained models. MobileNetV2 ImageNet weights are downloaded automatically by Keras when required.

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/CNN-CIFAR10-Image-Classification.git
```

### 2. Navigate to the Project Directory

```bash
cd CNN-CIFAR10-Image-Classification
```

### 3. Install Required Dependencies

```bash
pip install tensorflow scikit-learn scipy pandas numpy matplotlib seaborn jupyter
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 5. Execute the Notebook

Open `DL_PR4A.ipynb` and execute the cells sequentially.

The CIFAR-10 dataset and pretrained MobileNetV2 weights will be downloaded when required.

**Hardware Recommendation:** A GPU is recommended for faster CNN training and transfer-learning experiments. Training on a CPU may take considerably longer.

## Key Learnings

Through this project, the following Deep Learning concepts are explored:

- Convolutional Neural Networks.
- Convolution Operations.
- Padding and Strides.
- Feature Maps and Feature Extraction.
- MaxPooling and AveragePooling.
- Flatten and GlobalAveragePooling.
- LeNet-5 Architecture.
- Custom CNN Architecture Design.
- Forward Propagation and Backpropagation.
- Gradient Flow in CNNs.
- EarlyStopping and Learning Rate Scheduling.
- Image Data Augmentation.
- Overfitting and Generalization.
- Transfer Learning.
- Feature Extraction and Fine-Tuning.
- MobileNetV2 Architecture.
- Residual Connections.
- Multi-Class Image Classification.
- Model Performance Evaluation.

## Conclusion

This project provides a practical understanding of Convolutional Neural Networks and their application in multi-class image classification.

Starting with the CIFAR-10 dataset, the project progresses through fundamental convolution operations, pooling techniques, CNN architecture development, and model training.

It further explores data augmentation as a method of improving generalization and transfer learning through MobileNetV2 to leverage pretrained ImageNet features.

Finally, different CNN configurations are compared using test accuracy, parameter counts, and training performance to understand the trade-offs between model complexity, computational cost, and classification effectiveness.

The project demonstrates how modern Deep Learning techniques can be applied to image classification problems and provides a foundation for exploring more advanced architectures such as ResNet, EfficientNet, and other pretrained CNN models.

---

**Project Domain:** Deep Learning | Computer Vision | CNN | Image Classification | Transfer Learning

**Technologies:** Python | TensorFlow | Keras | Scikit-learn | NumPy | Pandas | Matplotlib | Seaborn
