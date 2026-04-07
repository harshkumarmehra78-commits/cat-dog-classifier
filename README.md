
---
title: Cat Dog
emoji: 🏆
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.11.0
app_file: app.py
pinned: false
---

# 🐱 Cat vs Dog Image Classifier

A comprehensive machine learning project for binary image classification that distinguishes between cats and dogs using multiple ML algorithms and deep learning techniques.

## 📋 Project Overview

This project implements and compares multiple machine learning models to classify images of cats and dogs:

- **Logistic Regression** - Baseline classical ML approach
- **Support Vector Machine (SVM)** - Advanced classical ML
- **Random Forest** - Ensemble classical ML method
- **Convolutional Neural Network (CNN)** - Deep learning with TensorFlow/Keras

The best-performing model (CNN) is deployed as an interactive web application using Gradio, available on Hugging Face Spaces.

## 🎯 Features

- ✅ **Multi-Model Comparison**: Compare performance across different algorithms
- ✅ **Deep Learning CNN**: Advanced image classification using transfer learning
- ✅ **Web Interface**: Interactive Gradio application for real-time predictions
- ✅ **Cloud Deployment**: Hosted on Hugging Face Spaces
- ✅ **Model Evaluation**: Comprehensive metrics (accuracy, precision, recall, F1-score)
- ✅ **Data Visualization**: Confusion matrices and performance comparisons

## 🏗️ Project Structure

```
cat-dog-final/
├── app.py                        # Gradio web interface (main deployment file)
├── cats_vs_dogs_classifier.py    # Complete ML pipeline & model training
├── cats_dogs_cnn.keras           # Pre-trained CNN model (serialized)
├── utils.py                      # Utility functions (frame extraction, etc.)
├── requirements.txt              # Project dependencies
├── README.md                     # Project documentation
└── .gitattributes               # Git attributes for large files
```

## 🔄 Workflow & Architecture

### Phase 1: Data Preparation

1. **Dataset Download**: Uses KaggleHub to fetch "Cats and Dogs Image Classification" dataset
2. **Data Organization**: Separates images into cat and dog categories
3. **Data Splitting**: 80% training, 20% testing data

### Phase 2: Model Training

#### Traditional ML Models:

- Images converted to feature vectors (flattened)
- Standardization using StandardScaler
- Models trained on extracted features:
  - Logistic Regression
  - SVM (Support Vector Machine)
  - Random Forest Classifier

#### Deep Learning CNN:

- Input image size: 224×224 pixels (RGB)
- Architecture: Convolutional layers → MaxPooling → Dense layers
- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Augmentation: ImageDataGenerator for training data
- Callbacks: EarlyStopping & ReduceLROnPlateau to prevent overfitting
- Epochs: 50 (with early stopping)

### Phase 3: Model Evaluation

- **Metrics Calculated**:
  - Accuracy Score
  - Precision, Recall, F1-Score
  - Confusion Matrix
  - Classification Report
- **Visualization**:
  - Model comparison charts
  - Confusion matrices heatmaps
  - Performance metrics display

### Phase 4: Deployment

- Best model (CNN) exported to `.keras` format
- Wrapped in Gradio interface for user interaction
- Deployed on Hugging Face Spaces platform

## 📦 Dependencies

```
gradio>=4.0.0          # Web interface framework
tensorflow>=2.13.0     # Deep learning library
numpy                  # Numerical computations
Pillow                 # Image processing
scikit-learn           # Traditional ML models
matplotlib             # Data visualization
seaborn                # Statistical visualization
opencv-python (cv2)    # Computer vision utilities
kagglehub              # Dataset download
```

## 🚀 Getting Started

### Local Setup

1. **Clone the Repository**

```bash
git clone <repository-url>
cd cat-dog-final
```

2. **Create Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure Kaggle API** (for dataset download)

```bash
# Download kaggle.json from Kaggle account settings
# Place it in ~/.kaggle/kaggle.json (or %USERPROFILE%\.kaggle\kaggle.json on Windows)
```

### Running the Training Pipeline

To train all models and compare performance:

```bash
python cats_vs_dogs_classifier.py
```

This will:

- Download the dataset from Kaggle
- Preprocess and split the data
- Train all four models
- Generate evaluation metrics and visualizations
- Save the best CNN model as `cats_dogs_cnn.keras`

### Running the Web App Locally

To test the Gradio interface:

```bash
python app.py
```

The interface will be available at `http://localhost:7860`

## 📊 Configuration Parameters

Key parameters in `cats_vs_dogs_classifier.py`:

| Parameter   | Value      | Description                               |
| ----------- | ---------- | ----------------------------------------- |
| IMG_SIZE    | (224, 224) | Image resize dimensions                   |
| CHANNELS    | 3          | RGB color channels                        |
| BATCH_SIZE  | 16         | Samples per gradient update               |
| EPOCHS      | 50         | Training iterations (with early stopping) |
| TEST_SIZE   | 0.2        | 20% test data split                       |
| RANDOM_SEED | 42         | Reproducibility seed                      |

## 💻 Main Components

### `app.py` - Web Interface

- Loads the trained CNN model
- Provides Gradio interface for image upload
- Handles image preprocessing (224×224 resize)
- Returns prediction with confidence scores
- Multiple fallback strategies for model loading
- Includes comprehensive logging for debugging

**Features:**

- Real-time image classification
- Confidence score display
- Error handling and recovery
- Support for multiple image formats (JPEG, PNG, BMP)

### `cats_vs_dogs_classifier.py` - Training Pipeline

- Orchestrates entire ML workflow
- Implements 4 different ML algorithms
- Performs hyperparameter configuration
- Generates comparative performance analysis
- Creates visualization plots
- Exports the best model

**Output:**

- Trained models
- Performance metrics CSV
- Visualization plots
- Model comparison report

### `utils.py` - Utility Functions

- `extract_frames()`: Extracts frames from video files
- Resizes images to 32×32 format
- Normalizes frame values (0-1 range)
- Handles video buffer padding

## 🎮 Using the Deployed App

The app is available on Hugging Face Spaces:

1. **Upload an Image**: Click the upload area or drag-and-drop
2. **Submit**: Click the "Submit" button
3. **Get Prediction**: View classification result and confidence

**Supported Formats**: JPEG, PNG, BMP, and other common image formats

## 📈 Expected Results

After training on the full dataset:

- **CNN Accuracy**: ~95-97% (typically the best performer)
- **Training Time**: ~10-20 minutes (depends on hardware)
- **Model Size**: ~50-100 MB

## 🔧 Troubleshooting

### Model Loading Issues

- Ensure `cats_dogs_cnn.keras` is in the project root
- Check file size (should be >10 MB for trained model)
- Verify TensorFlow version compatibility (>=2.13.0)

### Dataset Download Fails

- Configure Kaggle API credentials
- Check internet connection
- Use manual download if KaggleHub fails

### Memory Issues

- Reduce BATCH_SIZE in configuration
- Process fewer images for testing
- Use GPU if available (CUDA support)

## 🏆 Performance Metrics

The CNN model is evaluated using:

- **Accuracy**: Overall correctness rate
- **Precision**: True positive rate among predictions
- **Recall**: Detection rate of actual cats/dogs
- **F1-Score**: Harmonic mean of precision and recall

## 🛠️ Technologies Used

- **TensorFlow/Keras**: Deep learning framework
- **Scikit-learn**: Classical ML algorithms
- **Gradio**: Web interface framework
- **Hugging Face Spaces**: Cloud deployment platform
- **KaggleHub**: Dataset management
- **Pandas/NumPy**: Data manipulation
- **Matplotlib/Seaborn**: Data visualization

## 📝 Development Notes

- Uses MobileNetV2 architecture principles (224×224 input)
- Binary crossentropy loss for binary classification
- Adam optimizer for fast convergence
- Early stopping to prevent overfitting
- Data augmentation for improved generalization

## 🚢 Deployment Notes

The project is deployed on Hugging Face Spaces with:

- `app.py` as the main entry point
- `cats_dogs_cnn.keras` included in Git LFS
- Automatic containerization and hosting
- Real-time inference without server setup

## 📄 License

This project is provided as-is for educational and development purposes.

## 🤝 Contributing

To extend this project:

1. Improve model architecture
2. Add data augmentation techniques
3. Implement ensemble methods
4. Optimize for faster inference
5. Add support for other animal classifications

---

**Last Updated**: April 2026
**Status**: ✅ Active & Deployed
