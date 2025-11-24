# Pirate Pain – Multivariate Time-Series Classification (AN2DL Challenge 1)

This repository contains the full solution developed for Challenge 1 (A.Y. 2025/2026) of the **Artificial Neural Networks and Deep Learning** course at Politecnico di Milano.

The task focuses on **multivariate time-series classification** using the *Pirate Pain* dataset.  
Each sample consists of **160 time steps** with joint motion data, rule-based pain indicators, and subject metadata.  
The goal is to classify each sequence into three pain levels:

- **no_pain**
- **low_pain**
- **high_pain**

All models were **trained from scratch**, as required by the challenge (no pretrained models allowed).

Our team achieved **5.5 / 5 points** (full score + laude for top-performing groups).

---

## 📁 Folder Structure

### 📘 Notebooks

**`GroupKFold_Tuning.ipynb`**  
Performs **10-fold GroupKFold cross-validation** on all considered models during the hyperparameter tuning phase.  
Used to evaluate multiple configurations and select the best-performing ones.

**`notebook.ipynb`**  
Trains the **best selected models** (MLP and Ensemble) on the full training set and generates the final **Kaggle submission**.

---

### ⚙️ Scripts

**`run.sh`**  
Utility script that **trains all individual models** and generates their predictions.  
Must be executed **before running the final ensemble**.

**`script.py`**  
Main Python script from which `notebook.ipynb` was derived.  
Contains preprocessing, model definitions, training pipelines, and evaluation code.

**`scriptGroupKfold.py`**  
Base script used to create `GroupKFold_Tuning.ipynb`.  
Implements GroupKFold evaluation, hyperparameter tuning loops, and metric computation.

---

## 📂 Additional Folders

**`best_params/`**  
Stores the **best hyperparameter configurations** found during tuning.

**`report/`**  
Contains the **official project report** submitted for evaluation.

---

## 👥 Authors

**Tommaso Felice Banfi**  

**Francesco Seracini**  

**Matteo Lombardi**  

**Riccardo Ferro**  


