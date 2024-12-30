# Harmoni Fusion: A Multi-Model Ensemble for Infant Cry Intelligence

## Description
This repository contains the implementation of **Harmoni Fusion**, a novel framework designed to classify infant cries into distinct categories (e.g., burping, discomfort, hunger, tiredness, and belly pain). By combining the strengths of multiple deep learning models, this project aims to enhance the accuracy and robustness of infant cry classification, offering a scalable and effective solution for real-time applications in pediatric healthcare.

## Key Features
- **Hybrid Model Architecture**:
  - **2D CNN** trained on Mel spectrograms for spatial frequency feature extraction.
  - **1D CNN** trained on Mel Frequency Cepstral Coefficients (MFCCs) for temporal feature extraction.
  - Outputs of both models are integrated into a **Random Forest meta-classifier**.
- **Data Augmentation Techniques**:
  - Gaussian noise addition.
  - Random speed and pitch adjustments.
- **Performance Evaluation**:
  - Metrics include precision, recall, F1-score, and confusion matrices.
  - Demonstrates superior performance in noisy, real-world conditions.

## Dataset
The project utilizes the **Donate-a-Cry Corpus**, a benchmark dataset of infant cry recordings. To address data scarcity and imbalance, extensive data augmentation techniques were applied, resulting in a balanced and diverse dataset for training and evaluation.

## Methodology
1. **Data Preprocessing**:
   - Organizing raw audio files by class.
   - Applying augmentation techniques to balance the dataset.
2. **Feature Extraction**:
   - Mel spectrograms for 2D CNNs.
   - MFCCs for 1D CNNs.
3. **Model Development**:
   - Separate 2D and 1D CNN models optimized with dropout, pooling, and regularization techniques.
   - Integration of model predictions into a Random Forest meta-classifier.
4. **Training & Optimization**:
   - Hyperparameter tuning.
   - Class weights to handle imbalanced datasets.
5. **Evaluation**:
   - Tested on unseen data using performance metrics to validate robustness and accuracy.

## Results
The meta-model ensemble achieved the following recall values for the five classes:
- **Hungry**: **0.79** (Highest recall, indicating strong performance in detecting hunger-related cries).
- **Belly Pain**: **0.75** (High recall, showing accurate identification of belly pain cries).
- **Burping**: **0.72** (Moderate recall, with room for improvement).

These results demonstrate the framework's robustness, particularly for classes with distinct audio patterns like `hungry` and `belly pain` ,`burping`. However, challenges remain for classes with overlapping features like `tired`.

## Future Work
- Address dataset limitations by diversifying and expanding the data.
- Explore advanced feature extraction methods.
- Validate the framework in real-world scenarios.

## Installation and Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Harmoni-Fusion.git

## Install required  dependencies
``` bash
  pip install -r requirements.txt

