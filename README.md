# Breast Cancer Dataset Analysis and Classification

This project provides a professional analysis of a breast cancer dataset, focusing on building a machine learning model for patient status classification. The workflow includes data preprocessing, visualization, and implementing a Random Forest Classifier with hyperparameter optimization.

---

## Table of Contents

1. [Dataset Overview](#dataset-overview)
2. [Dependencies](#dependencies)
3. [Data Preprocessing](#data-preprocessing)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Model Training and Evaluation](#model-training-and-evaluation)
6. [Results](#results)
   
---

## Dataset Overview

The dataset `BRCA.csv` contains clinical and protein expression data for breast cancer patients. Key columns include:

- **Patient_ID**: Unique identifier for each patient
- **Age**: Patient's age
- **Protein1, Protein2, Protein3, Protein4**: Protein expression levels
- **Tumour_Stage**: Tumor stage (I, II, III)
- **Histology**: Histological subtype of the tumor
- **ER/PR/HER2 status**: Hormone receptor status
- **Surgery_type**: Type of surgery performed
- **Patient_Status**: Whether the patient is Alive or Dead

---

## Dependencies

This project requires the following Python libraries:

```python
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
```

Install them via:

```bash
pip install pandas matplotlib seaborn scikit-learn imbalanced-learn
```

---

## Data Preprocessing

1. **Duplicate Removal**: Identified and dropped duplicate rows.
2. **Handling Missing Values**: Dropped rows with missing values.
3. **Feature Encoding**:
    - Applied `LabelEncoder` for categorical variables such as `HER2 status`, `Surgery_type`, and `Patient_Status`.
    - Used `OrdinalEncoder` for the `Tumour_Stage` column.
    - Created one-hot encoding for the `Histology` column.
4. **Feature Selection**: Dropped unnecessary columns like `Patient_ID`, `Gender`, and dates.
5. **Class Balancing**: Used NearMiss under-sampling to handle class imbalance.

---

## Exploratory Data Analysis

### Visualizations:

1. **Patient Status Distribution**: 
   A bar plot showing the distribution of alive vs. dead patients.

2. **Age Distribution by Patient Status**: 
   A histogram illustrating the age distribution for each patient status.

3. **Tumour Stage Distribution by Patient Status**: 
   A grouped bar chart showing tumor stage distribution for each patient status.

4. **Tumor Stage Distribution by Age Group**: 
   A heatmap displaying tumor stage counts across different age groups.

5. **Protein Expression Levels**: 
   KDE plots for the distribution of Protein1, Protein2, Protein3, and Protein4 levels.

---

## Model Training and Evaluation

### Steps:

1. **Train-Test Split**:
   Split the resampled dataset into training and testing sets (80-20 split).

2. **Model**: 
   Used a Random Forest Classifier.

3. **Hyperparameter Tuning**: 
   Performed a grid search with 5-fold cross-validation to find the best combination of hyperparameters:

   - `n_estimators`: [500, 700]
   - `max_depth`: [10, 15, 20]
   - `min_samples_split`: [4, 6, 8]
   - `min_samples_leaf`: [2, 3, 4]
   - `max_features`: ["sqrt", "log2", 0.5]
   - `class_weight`: ["balanced", None]

4. **Evaluation Metrics**:
   - Precision
   - Recall
   - F1-Score
   - Accuracy

---

## Results

### Optimal Hyperparameters:

- `class_weight`: balanced
- `max_depth`: 10
- `max_features`: 0.5
- `min_samples_leaf`: 3
- `min_samples_split`: 4
- `n_estimators`: 500

### Classification Report:

#### Training Set:

| Metric       | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Alive (0)    | 0.94      | 1.00   | 0.97     | 47      |
| Dead (1)     | 1.00      | 0.94   | 0.97     | 52      |
| **Accuracy** |           |        | 0.97     | 99      |

#### Test Set:

| Metric       | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Alive (0)    | 0.88      | 1.00   | 0.94     | 15      |
| Dead (1)     | 1.00      | 0.80   | 0.89     | 10      |
| **Accuracy** |           |        | 0.92     | 25      |

---

## Acknowledgments

This project utilizes the `BRCA.csv` dataset for academic purposes, showcasing advanced data analysis and machine learning methodologies.

