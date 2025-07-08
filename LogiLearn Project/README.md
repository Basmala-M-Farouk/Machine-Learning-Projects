# ⚙️ OptiLogiX  
**Optimizer Strategies for Logistic Regression in Binary Classification**

## 📘 Overview

This project implements **Logistic Regression from scratch** in NumPy and benchmarks three popular optimization techniques:

- ✅ Gradient Descent (GD)  
- 🔄 Stochastic Gradient Descent (SGD)  
- 🔀 Mini-Batch Gradient Descent  

The goal is to compare their performance in terms of:
- Convergence speed
- Classification metrics (Accuracy, Precision, Recall, F1-score)
- Training stability

---

## 📂 Dataset

The dataset is synthetically generated using `sklearn.datasets.make_classification()` to simulate a real-world binary classification scenario with:
- 10,000 samples
- 30 features (15 informative)

---

## 📊 Evaluation Metrics

Each optimizer is evaluated using:
- 🔹 Confusion Matrix
- 🔹 Accuracy
- 🔹 Precision
- 🔹 Recall
- 🔹 F1-Score

---

## 🚀 How to Run

```bash
1. Clone the repo or open the notebook in Google Colab
2. Run all cells in order

