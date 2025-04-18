# BMI Prediction and Gender Classification from Facial Images

## Introduction
This project explores the use of machine learning and deep learning techniques for facial feature analysis using front and side-profile mugshots. It predicts BMI and Gender from extracted image features and also analyzes offense trends from a dataset. It leverages **VGG16**, **MTCNN**, **PCA**, and **SVM/Linear Regression** to build an end-to-end pipeline for pattern recognition and classification tasks. The 
dataset used for training is **Illinois DOC Dataset**.

## 🔍 Process Overview

### Face Detection:
- MTCNN is used to detect and crop faces from both front and side images.

### Feature Extraction:
- VGG16 (pretrained on ImageNet) is used to extract 4096-dimensional features from each image.

### Feature Selection & Preprocessing:
- Zero-only columns and low-variance columns are dropped.
- PCA is used for dimensionality reduction specific to each prediction task (BMI and Gender).

### Model Training:
- **BMI Prediction**: Linear Regression.
- **Gender Classification**: Custom and built-in SVM.

## Key Features
- Front and Side face detection using MTCNN.
- Deep feature extraction using VGG16.
- Dual-model prediction for BMI (regression) and Gender (classification).
- Efficient feature reduction using PCA tailored to each target variable.
- Clean data pipeline with modular structure.
- Visual analytics for offense distribution.


## 📚 Learnings and Improvements

### Learnings:
- **Feature Extraction with VGG16**: Utilizing VGG16 for feature extraction was a valuable learning experience. It helped in understanding how powerful pre-trained deep learning models can be for extracting complex image features. The 4096-dimensional features extracted provided a rich representation, though further optimization might enhance their relevance to the BMI and Gender prediction tasks.
- **Dimensionality Reduction with PCA**: Applying PCA to reduce the high-dimensional feature space provided useful insights into the importance of dimensionality reduction. This process helped highlight the most relevant features for prediction tasks, demonstrating the importance of selecting the right number of components for each model.
- **Gender Classification & Regression**: Using both SVM and Linear Regression gave hands-on experience in comparing a classification model (SVM) with a regression model (Linear Regression), helping to understand the trade-offs between these two approaches for different types of prediction tasks.
- **Dataset Handling**: While working with the dataset, it became clear how crucial proper data preprocessing and handling are for any machine learning task. Managing data variability and ensuring quality input data are fundamental steps in building an effective model.

### Possible Improvements:
- **Addressing Dataset Imbalance**: One of the major challenges faced during this project was the imbalance in the dataset, especially for the gender classification task. To improve accuracy, techniques such as oversampling the minority class (e.g., SMOTE), undersampling the majority class, or using class weights in the model could be implemented.
- **Evaluation Metrics**: Implementing more comprehensive evaluation metrics, such as Precision, Recall, F1-score, and ROC-AUC, would provide deeper insights into model performance, especially when handling imbalanced datasets.
- **Augmenting Data Diversity**: Expanding the dataset to include a more diverse range of facial features and profiles could help improve the generalizability of the model and yield more reliable predictions.


