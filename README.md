> [!WARNING]
> This README is outdated as of the latest commit. WIP to update the README

# ZTF Asteroid Detection Pipeline

A machine learning pipeline for detecting asteroids in Zwicky Transient Facility (ZTF) alert data using feature extraction and XGBoost classification.

## Overview

This project processes ZTF astronomical alert data to identify asteroid candidates through difference imaging and machine learning. The pipeline:  
- Processes Avro format alerts from ZTF tarballs
- Extracts science and template cutouts and processes the difference cutout
- Identifies potential asteroid candidates using blob detection
- Generates positive and negative samples
- Trans an XGBoost classifier with HOG features and statistical measures
- Evaluated model performance with comprehensive metrics

## Pipeline Architecture

```
ZTF Tarballs -> Alert Parsing -> Image Preprocessing -> Difference Imaging -> Blob Detection -> Feature Extraction -> Model Training -> Evaluation
```

## Project Structure

```bash
├── main.py # Main execution script
├── retrieval.py # AVRO alert parsing and cutout decoding
├── preprocessing.py # Image processing and blob detection
├── dataset_builder.py # Feature extraction and dataset creation
├── train_model.py # Model training and evaluation
├── output_log.txt # Example execution output
└── ztf_pipeline_output/ # Generated datasets and model files
```

## Key Features

### Data processing
- **Alert Decoding**: Parsing ZTF Avro alerts from compressed tarballs
- **Image Preprocessing**: Median filtering and normalization
- **Difference Imaging**: Science vs template image subtraction
- **Blob Detection**: Identifies significant transient sources

### Feature Extraction
- **Statistical Features**: Mean and standard deviation of difference patches
- **HOG Features**: Histogram of Oriented Gradients (12 orientations)
- **Smart Negative Sampling**: Background patches away from central sources

![Sample Images](https://github.com/Negative-Zero-Official/Asteroid-Detection-using-Machine-Learning/blob/3279d6a31e7a8728b546ae412169441eea25a56a/ztf_pipeline_output/Figure_1.png)

### Machine Learning
- **XGBoost Classifier**: GPU-accelerated training on an NVIDIA RTX 5070 GPU
- **Group-based Splitting**: Prevents data leakage between alerts
- **Comprehensive Evaluation**: Precision, recall, F1-score, confusion matrix

## Model Performance

### Training Results
The model achieves excellent performance on the test set:

```plaintext
Precision: 0.988 Recall: 0.991 F1: 0.990
Accuracy: 99% on 103,619 test samples
```

### Confusion Matrix

![Confusion Matrix](https://github.com/Negative-Zero-Official/Asteroid-Detection-using-Machine-Learning/blob/3279d6a31e7a8728b546ae412169441eea25a56a/ztf_pipeline_output/confusion_matrix.jpg)

## Usage

1. **Prepare Data**: Place ZTF tarballs in the `tarballs/` directory
2. **Run Pipeline**: Execute the main script
    ```bash
    python main.py
    ```
3. **Monitor Progress**: Check `output_log.txt` for detailed output
4. **Results**: Find trained model and datasets in `ztf_pipeline_output/`

## Dataset Statistics

- **Total Samples**: 518,284
- **Features**: 14 total (2 statistical + 12 HOG features)

## Model Details

- **Algorithm**: XGBoost with GPU acceleration
- **Objective**: Binary logistic regression
- **Evaluation**: Log loss with early stopping
- **Validation**: Group-based split by alert ID to prevent leakage