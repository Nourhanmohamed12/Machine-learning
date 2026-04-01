Cats vs Dogs Image Classification with CNN & Hyperparameter Tuning

🎯 Overview
This project implements a Convolutional Neural Network (CNN) for binary image classification between Cats and Dogs using advanced techniques:

Data augmentation for improved generalization
Keras Tuner for automated hyperparameter optimization
Transfer learning ready architecture
Comprehensive evaluation pipeline
Visualization of predictions and hyperparameters
Achieves ~95.7% training accuracy and ~72.7% validation accuracy with data augmentation.

📊 Dataset
Classes: Cats (9,999 samples), Dogs (10,000 samples)
Image Size: 150x150 grayscale
Total Samples: ~20,000 images
Train/Validation Split: 70/30 (13,999 train, 6,000 validation)
Source: Custom AnimalImages dataset

Dataset Split:
├── Training: 13,999 images (150×150×1)
└── Validation: 6,000 images (150×150×1)

✨ Features
✅ Image preprocessing (grayscale, resize, normalization)
✅ Data augmentation (rotation, zoom, shift, flip)
✅ CNN with multiple conv-pool-dropout blocks
✅ Keras Tuner RandomSearch hyperparameter optimization
✅ Learning rate reduction on plateau
✅ Prediction visualization with confidence scores
✅ Hyperparameter performance visualization
✅ Model evaluation & comparison

🏗️ Architecture
Base CNN Model

Layer (type)              Output Shape       Param #
Conv2D (12, 3×3)         (150, 150, 12)     120
MaxPool2D (2×2)          (75, 75, 12)       0
Conv2D (24, 3×3)         (75, 75, 24)       2,616
MaxPool2D (2×2)          (37, 37, 24)       0
Flatten                  (33,528)           0
Dense (512)              (512)              17,154,528
Dropout (0.1)            (512)              0
Dense (1, sigmoid)       (1)                513
Total params: 17,157,777

Tuned Model Features

Hyperparameters Tuned:
├── conv_layers: 1-5
├── filter_num: 2-24 (step=2)
├── hidden_layers: 1-5
├── layer_units: 156-1048
├── conv_dropout: 0.05-0.2
└── hidden_dropout: 0.05-0.2

🚀 Installation
Prerequisites

Python 3.8+, TensorFlow 2.x, Keras Tuner

1. Clone Repository

git clone https://github.com/Nourhanmohamed12/machine-project(2).git
cd cats-dogs-cnn

2. Setup Environment

# Create conda environment
conda create -n catdog python=3.9
conda activate catdog

# Install dependencies
pip install -r requirements.txt
requirements.txt

tensorflow>=2.10.0
keras-tuner>=1.3.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
scikit-learn>=1.2.0
Pillow>=9.0.0
opencv-python>=4.7.0

📖 Usage

1. Prepare Dataset

data/
├── cats/          # ~10,000 cat images (.jpg, .png, .jpeg)
└── dogs/          # ~10,000 dog images (.jpg, .png, .jpeg)

2. Run Training Pipeline
   
# Single model training
python src/train_base_model.py

# Hyperparameter tuning
python src/tune_hyperparameters.py --trials 10

# Full pipeline with evaluation
python src/main.py

3. Jupyter Notebook
   
jupyter notebook notebooks/cats_dogs_analysis.ipynb

📈 Results
Training Progress

Epoch 1/10: loss=0.8420, accuracy=56.77%
Epoch 5/10: loss=0.3970, accuracy=82.03%
Epoch 10/10: loss=0.1349, accuracy=95.74% ⭐

Validation Performance

Validation Results:
├── Accuracy: 72.70%
├── Loss: 0.7653
└── Prediction Visualization: ✅

Hyperparameter Analysis

Best Configuration:
├── conv_layers: 3
├── filter_num: 12
├── hidden_layers: 2
├── layer_units: 756
└── val_accuracy: 72.60%

🔧 Hyperparameter Tuning
Keras Tuner Integration
tuner = keras_tuner.RandomSearch(
    CNN_model_tuner,
    objective='val_accuracy',
    max_trials=10,
    project_name="cat_dog_CNN"
)
Tuned Parameters Visualized

[Scatter plots showing filter_num, conv_layers, hidden_layers vs accuracy]
🎨 Prediction Visualization

8×10 Grid showing:
├── Predicted class & confidence (50-100%)
├── Color-coded: Green(✅), Orange(⚠️), Red(❌)
├── Confidence scaling for intuitive understanding

👩‍💻 Author

Nourhan Mohammed
Computer Science Student | Data Enthusiast
