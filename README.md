# 💳 Credit Card Fraud Detection

This end-to-end machine learning project detects fraudulent credit card transactions using both **supervised** and **unsupervised** models. It handles real-world challenges like **class imbalance** and emphasizes metrics such as **Recall** and **F1-Score** to minimize false negatives.

---

## 📌 Objective

To build, evaluate, and compare different machine learning models for identifying credit card fraud by leveraging:
- Supervised classifiers
- Unsupervised anomaly detection methods
- Class balancing techniques (e.g., SMOTE)

---

## 📂 Dataset Overview

- **Source**: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Fraudulent**: 492 (0.17%)
- **Features**: 30 (PCA-transformed V1-V28, Time, Amount)
- **Target**: `Class` (0 = Legit, 1 = Fraud)

---

## 🔍 Workflow Breakdown

### 1. Data Exploration & Preprocessing
- Null check, distribution analysis, scaling via `RobustScaler`
- Visualizations for feature relationships, class imbalance

### 2. Unsupervised Anomaly Detection 🧪
Used before labels are known, treating frauds as anomalies:
- **KMeans Clustering**: Unsupervised classification based on cluster distance
- **Isolation Forest**: Tree-based anomaly detection focusing on isolation paths

> These models are evaluated using confusion matrix and precision/recall after mapping predictions against true labels.

### 3. Supervised Learning Models 🤖

| Model                   | Dataset        | Key Notes                                 |
|------------------------|----------------|--------------------------------------------|
| Gaussian Naive Bayes   | Imbalanced + SMOTE | Simple baseline, fast                      |
| Logistic Regression    | Imbalanced + SMOTE | Strong baseline with regularization       |
| Decision Tree Classifier | Imbalanced + SMOTE | Interpretable, overfits easily            |

- Evaluated on both original & balanced datasets using **SMOTE**
- Metrics used: Accuracy, Precision, Recall, F1-Score, ROC-AUC

---

## 📈 Model Performance Summary

- **Imbalanced models** → high accuracy, low recall (miss frauds)
- **SMOTE models** → improved recall and balanced performance
- **Unsupervised models** → decent at isolating anomalies without training labels

---

## 📊 Evaluation Focus

| Metric    | Why It Matters                        |
|-----------|----------------------------------------|
| **Recall** | Minimize false negatives (missed fraud) |
| **F1-Score** | Balances precision and recall         |
| **Confusion Matrix** | Understand classification errors |
| **ROC-AUC** | Compare classifier discrimination ability |

---

## 📦 Libraries Used

```bash
numpy
pandas
matplotlib
seaborn
scikit-learn
imblearn
scipy
```



---

## 👤 Author

**Shrey Shah**  
