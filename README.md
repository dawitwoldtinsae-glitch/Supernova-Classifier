# Supernova Classifier

A machine learning project for classifying astronomical transients as **Supernova (SN)** or **Non-Supernova (Non-SN)** using observational and light-curve-derived features.

The project follows a complete machine learning workflow, starting from raw astronomical data exploration and cleaning, through feature engineering and baseline modeling, to hyperparameter tuning, validation, and final model selection.

---

## Project Goal

Astronomical transient surveys contain many objects whose brightness changes over time. Identifying which of these objects are supernovae is an important task in time-domain astronomy.

The goal of this project is to build a supervised binary classification model that predicts whether an astronomical transient is a supernova.

The target variable is:

- **1 — Supernova (SN)**
- **0 — Non-Supernova (Non-SN)**

This is **Version 1** of the project. The purpose of this version is to establish a complete and reproducible machine learning pipeline using the available dataset. Future versions can improve the model by incorporating additional observations, features, and larger datasets.

---

## Dataset

The project uses astronomical transient data containing object classifications and light-curve observations.

The final modeling dataset contains:

- **3,734 astronomical objects**
- **15 predictive features**
- **1 binary target variable**
- **1,449 Supernova objects**
- **2,285 Non-Supernova objects**

Objects with only a single light-curve observation were removed because several statistical variability features cannot be meaningfully calculated from a single observation.

The raw dataset is not included in the repository. Only the final cleaned dataset used for modeling is included.

### Final Dataset

`data/merge_all.csv`

---

## Features

The final model uses the following features:

| Feature | Description |
|---|---|
| `Dec (J2000)` | Declination of the astronomical object |
| `SDSS` | Whether SDSS information is available |
| `count` | Number of light-curve observations |
| `mag_info` | Magnitude information associated with the object |
| `mag_max` | Faintest observed magnitude |
| `mag_mean` | Mean observed magnitude |
| `mag_median` | Median observed magnitude |
| `mag_min` | Brightest observed magnitude |
| `mag_std` | Standard deviation of magnitude |
| `magerr_max` | Maximum magnitude measurement error |
| `magerr_mean` | Mean magnitude measurement error |
| `magerr_std` | Standard deviation of magnitude measurement error |
| `mjd_std` | Standard deviation of observation times |
| `baseline` | Time span between the earliest and latest observation |
| `mag_baseline` | Difference between maximum and minimum observed magnitude |

The original classification column was removed before modeling to prevent **target leakage**.

---

# Project Structure

```text
Supernova-Classifier/
│
├── data/
│   └── merge_all.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_baseline_models.ipynb
│   ├── 05_hyperparameter_tuning.ipynb
│   └── 06_final_model_comparison.ipynb
│
├── src/
│   ├── model.py
│   └── untiles.py
│
├── results/
│   ├── training_results.csv
│   └── test_results.csv
│
└── README.md
```

---

# Machine Learning Workflow

The project is divided into six main notebooks.

## 01 — Data Exploration

`notebooks/01_data_exploration.ipynb`

The first notebook investigates the original astronomical data and establishes an understanding of its structure.

Main tasks include:

- Loading the astronomical datasets
- Inspecting DataFrame structures
- Examining the light-curve observations
- Investigating object classifications
- Checking class distributions
- Exploring magnitude distributions
- Exploring magnitude-error distributions
- Examining observation-time distributions
- Investigating the number of observations per object
- Performing initial exploratory data analysis
- Identifying potential problems in the original data

The purpose of this notebook is to understand the data before making cleaning or modeling decisions.

---

## 02 — Data Cleaning

`notebooks/02_data_cleaning.ipynb`

This notebook prepares the astronomical data for feature engineering and modeling.

Main tasks include:

- Structuring the original data
- Cleaning classification labels
- Separating Supernova and Non-Supernova objects
- Removing unsuitable or problematic observations
- Examining class distributions after cleaning
- Investigating potentially ambiguous classifications
- Removing objects with insufficient observations
- Checking for missing values
- Producing a clean dataset for the next stage

Objects with only one observation were removed because variability statistics such as standard deviation are not meaningful with a single observation.

The resulting dataset contains **3,734 objects with no missing values**.

---

## 03 — Feature Engineering

`notebooks/03_feature_engineering.ipynb`

This notebook converts the cleaned astronomical observations into object-level machine learning features.

Because individual objects have multiple light-curve observations, statistical summaries are calculated for each object.

The engineered features include:

- Observation count
- Mean, median, minimum, and maximum magnitude
- Magnitude standard deviation
- Mean, median, minimum, maximum, and standard deviation of magnitude errors
- Observation-time statistics
- Observation baseline
- Magnitude baseline

Feature analysis was then used to determine which variables were useful enough to retain.

Some features were removed because they were redundant, weakly informative, or unlikely to contribute useful predictive information.

The final feature set was then prepared for machine learning.

---

## 04 — Baseline Models

`notebooks/04_baseline_models.ipynb`

This notebook establishes the initial performance of several machine learning algorithms before hyperparameter tuning.

The evaluated models are:

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Random Forest

The models are evaluated on both the training and held-out test sets.

Evaluation includes:

- Accuracy
- Precision
- Recall
- F1 score
- Macro F1
- ROC AUC
- Precision-Recall Average Precision
- Cohen's Kappa
- Confusion matrices
- ROC curves
- Precision-Recall curves

The baseline experiments showed that **Random Forest performed substantially better than the other baseline models on the held-out test set**.

---

## 05 — Hyperparameter Tuning

`notebooks/05_hyperparameter_tuning.ipynb`

This notebook investigates whether model performance can be improved through hyperparameter optimization.

### Logistic Regression

A polynomial Logistic Regression model was investigated using degree-2 polynomial features.

`GridSearchCV` was used to search over different regularization strengths.

The best value was:

- **C = 100**
- Cross-validation macro F1 ≈ **79.51%**

### Random Forest

Random Forest hyperparameters were also optimized using `GridSearchCV`.

The search included:

- Number of estimators
- Maximum tree depth
- Minimum samples per leaf

The best configuration was:

- **n_estimators = 200**
- **max_depth = 20**
- **min_samples_leaf = 1**

The best cross-validation macro F1 was approximately:

**80.24%**

Repeated stratified cross-validation was used to reduce dependence on a single data split.

---

## 06 — Final Model Comparison

`notebooks/06_final_model_comparison.ipynb`

The final notebook compares the strongest candidate models and performs the final model-selection analysis.

The main candidates were:

- Polynomial Logistic Regression
- Random Forest
- Soft Voting Classifier

The Soft Voting Classifier combined the Polynomial Logistic Regression model and Random Forest using their predicted probabilities.

Although the ensemble produced strong training results, Random Forest provided the best overall balance of performance, simplicity, and validation stability.

The final Version 1 model is therefore:

**Random Forest**

---

# Reusable Modules

The project also contains reusable Python modules so that model construction and evaluation code does not have to be duplicated throughout the notebooks.

## `src/model.py`

This module contains reusable model pipeline factories for:

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors
- Random Forest

Models that require feature scaling use scaling inside their pipelines.

This is important because preprocessing must be performed within the machine learning pipeline, particularly when cross-validation and hyperparameter tuning are used. It prevents information from the validation folds from leaking into the training process.

The functions in this module return **unfitted models**. Training, prediction, evaluation, and hyperparameter tuning remain the responsibility of the notebooks.

---

## `src/untiles.py`

This module contains reusable analysis and visualization utilities developed during the project.

The main utilities include:

### ROC Curve Plot

Creates a ROC curve and displays the model's ROC AUC.

### Precision-Recall Plot

Creates a Precision-Recall curve and reports the model's average precision.

### Confusion Matrix Visualization

Creates a confusion matrix with support for normalized values and customizable labels.

The project uses:

- **0 = Non-SN**
- **1 = SN**

### Classification Evaluation Plots

Combines the major classification visualizations into a single figure:

- ROC curve
- Confusion matrix
- Precision-Recall curve

These utilities make the model evaluation process consistent across the different notebooks.

---

# Model Selection & Validation

The final model selected was **Random Forest**, chosen based on repeated cross-validation performance rather than a single train/test split.

While Random Forest achieved **100% accuracy on the training set**, this reflects the algorithm's inherent tendency to fit training data very closely due to its use of deep, largely unpruned decision trees. This is expected behavior and is not evidence of a flawed pipeline.

To obtain a reliable estimate of real-world performance, the model was evaluated using **RepeatedStratifiedKFold cross-validation (3 splits, 5 repeats, 15 total folds)** after hyperparameter tuning via `GridSearchCV`, using **macro F1** as the optimization metric.

Across these folds, the model achieved:

| Metric | Mean | Standard Deviation |
|---|---:|---:|
| Accuracy | **81.45%** | ±1.22% |
| Macro F1 | **80.24%** | ±1.35% |
| SN Precision | **77.72%** | ±1.54% |
| SN Recall | **73.17%** | ±2.57% |
| SN F1 | **75.35%** | ±1.81% |

These cross-validated results are reported as the expected generalization performance rather than the single test-split metrics.

The single test split produced a somewhat more optimistic SN recall of **77.59%**, which fell near the upper end of the cross-validation range. Reporting the repeated cross-validation results therefore provides a more reliable picture of expected performance.

The relatively low standard deviations across folds indicate that the model's performance is reasonably stable and is not primarily the result of a lucky data split.

---

# Final Model Performance

The final comparison on the held-out test set was:

| Model | Accuracy | Macro F1 | SN Precision | SN Recall | SN F1 | ROC AUC | PR AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Polynomial LR (C=100) | 80.05% | 78.99% | 74.39% | 74.14% | 74.27% | 0.88 | 0.78 |
| Random Forest | **82.06%** | **81.16%** | **76.53%** | **77.59%** | **77.05%** | 0.89 | 0.79 |
| Soft Voting | 80.72% | 79.76% | 74.83% | 75.86% | 75.34% | **0.90** | **0.80** |

Random Forest achieved the strongest overall hard-classification performance while remaining considerably simpler than the ensemble approach.

The final model-selection decision is therefore based on the combination of:

- Strong classification performance
- Competitive ROC AUC
- Strong SN F1
- Stable repeated cross-validation performance
- Simpler model architecture
- No meaningful improvement from the more complex ensemble

---

# Evaluation Philosophy

Accuracy alone is not sufficient for evaluating this project.

The dataset contains both Supernova and Non-Supernova objects, and the two classes have different scientific implications. For this reason, the project considers several complementary metrics.

### Accuracy

Measures the overall proportion of correctly classified objects.

### Precision

For the SN class, precision answers:

> Of the objects predicted to be supernovae, how many actually are supernovae?

### Recall

For the SN class, recall answers:

> Of all actual supernovae, how many did the model successfully identify?

### F1 Score

Balances precision and recall.

### Macro F1

Calculates F1 independently for both classes and gives them equal importance. This makes it useful when evaluating performance across both SN and Non-SN classes.

### ROC AUC

Measures how well the model separates the two classes across different classification thresholds.

### Precision-Recall Average Precision

Provides another view of the model's ability to identify the positive SN class, particularly when considering the class distribution.

### Cohen's Kappa

Measures agreement between predictions and true labels while accounting for agreement that could occur by chance.

---

# Limitations

## SN Recall Variability

Of all evaluated metrics, SN recall showed the highest fold-to-fold variability:

**±2.57%**

with a cross-validation range of approximately:

**68.91%–77.98%**

This makes SN recall the least stable of the main evaluated metrics.

Since missing a true supernova (**false negative**) can be more costly than producing a false positive in transient classification, this metric should be monitored closely if the model is eventually deployed.

The reported approximately **73% SN recall should therefore be interpreted as an expected average**, not as a guaranteed minimum performance.

---

## Training Fit Is Not Representative

Training-set metrics, including the Random Forest's **100% training accuracy**, should not be used to judge the model's real-world quality.

The Random Forest uses deep decision trees that can fit the training observations very closely.

The appropriate reference for expected generalization is the repeated cross-validation performance:

- Accuracy: **81.45% ± 1.22%**
- Macro F1: **80.24% ± 1.35%**
- SN F1: **75.35% ± 1.81%**

These values provide a much more realistic estimate of how the model is expected to perform on unseen astronomical objects.

---

## Dataset Size

The final dataset contains 3,734 objects. While this is sufficient for establishing a meaningful Version 1 model, a larger and more diverse astronomical dataset could improve generalization.

Future versions should investigate whether additional transient observations and larger datasets improve the model's ability to distinguish difficult classes.

---

## Feature Limitations

The current model uses statistical summaries of the light curves rather than the complete time-series structure.

Important information about the detailed shape of a transient's light curve may therefore be lost.

Future versions could investigate features such as:

- Rise and decline rates
- Light-curve asymmetry
- Color information
- More detailed variability statistics
- Time-series features
- Additional astronomical measurements
- More sophisticated representations of the complete light curve

---

## Class Ambiguity

Astronomical transient classifications can sometimes be ambiguous or contain mixed/uncertain labels.

The Version 1 dataset focuses on producing a practical binary classification problem rather than attempting to resolve every possible astronomical subclass.

This means that some difficult or scientifically ambiguous objects may remain challenging for the classifier.

---

# Future Improvements

Potential Version 2 improvements include:

1. **Larger astronomical datasets**
2. **More complete light-curve information**
3. **Additional astronomical features**
4. **Time-series-specific feature engineering**
5. **Feature importance and interpretability analysis**
6. **Threshold optimization for SN recall**
7. **Comparison with additional machine learning algorithms**
8. **External validation on a separate astronomical dataset**
9. **Investigation of difficult SN/AGN cases**
10. **Deep learning approaches for raw light-curve data**

The project is intentionally structured so that new data and features can be incorporated without rebuilding the entire workflow from scratch.

---

# Technologies

The project was developed using:

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

Machine learning techniques used include:

- Logistic Regression
- Polynomial Features
- Decision Trees
- K-Nearest Neighbors
- Random Forest
- Soft Voting
- Standardization
- GridSearchCV
- Repeated Stratified Cross-Validation
- ROC analysis
- Precision-Recall analysis
- Confusion matrices

---

# Version 1 Conclusion

This project establishes a complete end-to-end machine learning pipeline for astronomical transient classification.

The workflow covers:

**Data exploration → Data cleaning → Feature engineering → Baseline modeling → Hyperparameter tuning → Cross-validation → Model comparison → Final model selection**

The final Version 1 model is a **Random Forest classifier**.

Its repeated cross-validation performance indicates an expected generalization accuracy of approximately **81.45%** and macro F1 of approximately **80.24%**, with an SN F1 of approximately **75.35%**.

The results provide a solid baseline for future versions of the project, particularly those incorporating richer astronomical observations and more informative representations of transient light curves.

---

# Author

Developed as a machine learning project to study the application of supervised classification to astronomical transient data.
