

# ML-Classification-Breast-Cancer-Prediction

This project uses machine learning classification algorithms to predict whether a tumor is benign or malignant based on various features extracted from breast cancer medical data. The goal is to demonstrate the application of EDA, preprocessing, and supervised ML models for binary classification problems in healthcare.

---

## 📁 Dataset

The dataset used is the **Breast Cancer Wisconsin (Diagnostic) Data Set**, available [here](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29).

* **Target Feature**: `diagnosis` (`M` = Malignant, `B` = Benign)
* **Input Features**: 30 numerical features including mean radius, texture, perimeter, area, smoothness, etc.
* **Total samples**: 569
* **Classes**: 2 (Malignant, Benign)

---

## 📊 Exploratory Data Analysis (EDA)

* Data inspection using `.head()`, `.info()`, `.describe()`
* Visualizing class distribution with histograms and bar plots
* Feature-wise comparison of diagnosis using `pd.crosstab()` and scatter plots
* Correlation matrix and heatmap to understand inter-feature relationships

---

## 🧼 Data Preprocessing

* Dropped unnecessary `id` column
* Handled missing values (None found)
* Label split:

  * **X**: Feature matrix (all columns except `diagnosis`)
  * **y**: Target variable (`diagnosis`)
* Train-test split (80/20) using `train_test_split`

---

## 🤖 Models Used

Three ML classifiers were tested:

* **Logistic Regression**
* **K-Nearest Neighbors (KNN)**
* **Random Forest Classifier**

---

## 🧪 Evaluation Strategy

* Accuracy scores on test data
* `fit_and_score()` function evaluates all models uniformly
* `model_compare` bar plot displays accuracy comparison
* Further scope for evaluating:

  * Confusion Matrix
  * Classification Report
  * ROC Curve
  * F1, Precision, and Recall Scores

---

## 📈 Results

A simple model benchmarking showed comparative test accuracies of:

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 0.95+    |
| KNN                 | \~0.91   |
| Random Forest       | \~0.96   |

(Note: Values may slightly vary with data shuffling.)

---

## 📌 Future Work

* Hyperparameter tuning using `GridSearchCV` or `RandomizedSearchCV`
* Feature scaling to improve KNN performance
* Model deployment using `Streamlit` or `Flask`
* Add confusion matrix and ROC curve plots
* Convert categorical labels into binary values for consistency

---

## 📚 Requirements

Install dependencies using:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```


