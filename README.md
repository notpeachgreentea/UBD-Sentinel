# 🛡️ Sentinel: Insider Threat Detection Using Behavioral Analytics

## 📌 Overview
Sentinel is a machine learning-based insider threat detection system that analyzes user behavioral patterns to identify potential malicious activity. The system leverages a stacked Long Short-Term Memory (LSTM) model and compares its performance with traditional machine learning models using the CERT Insider Threat Dataset v4.2.

---

## 🎯 Objectives
- Detect insider threats based on user behavioral activity
- Compare deep learning (LSTM) with traditional ML models
- Address class imbalance using CTGAN (synthetic data generation)
- Evaluate performance using multiple metrics

---

## 📊 Dataset
- Source: CERT Insider Threat Dataset v4.2  
- Simulates ~1000 employees over 18 months  
- Includes various insider threat scenarios:
  - Data theft
  - Sabotage
  - Data leakage  

### Data Types:
- Logon activity  
- Email interactions  
- File access  
- Web browsing  
- Device (USB) usage  

> ⚠️ Note: Dataset is not included due to size and access restrictions.

---

## ⚙️ Features Used
The model uses 14 engineered behavioral features:

- logon_count  
- after_hours_count  
- usb_count  
- email_count  
- file_count  
- O, C, E, A, N (psychometric traits)  
- web_total_clicks  
- web_cloud_count  
- web_social_count  
- web_job_count  

---

## 🧠 Models Used

### Main Model:
- **Stacked LSTM (Proposed Model)**

### Baseline Models:
- Decision Tree  
- Random Forest  
- XGBoost  
- Support Vector Machine (Linear)  
- Logistic Regression  
- Isolation Forest  

---

## 📈 Results

### 🔥 Sentinel LSTM Performance:
- Accuracy: **99.90%**  
- Precision: **99.54%**  
- Recall: **98.25%**  
- F1-Score: **98.89%**

The LSTM model demonstrates strong performance while effectively capturing complex behavioral patterns.

---

## ⚖️ Comparative Insights
- Ensemble models (XGBoost, Random Forest) achieved near-perfect performance  
- LSTM provides better behavioral modeling capability  
- Traditional models (SVM, Logistic Regression) showed poor recall  
- Isolation Forest (unsupervised) performed significantly lower  

> ⚠️ Note: Extremely high accuracy may indicate potential overfitting due to synthetic data augmentation.

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
