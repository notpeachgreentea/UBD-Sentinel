"""
Sentinel: Insider Threat Detection Using Behavioral Analytics
Final model training and evaluation script
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense

import xgboost as xgb


# =========================
# CONFIGURATION
# =========================

DATA_PATH = "data/final_gold_dataset.csv"
MODEL_PATH = "models/insider_threat_model.h5"
SCALER_PATH = "models/std_scaler.bin"
RESULTS_PATH = "results/model_comparison_results.csv"
CONFUSION_MATRIX_PATH = "results/lstm_confusion_matrix.png"

FEATURES = [
    "logon_count", "after_hours_count", "usb_count", "email_count", "file_count",
    "O", "C", "E", "A", "N",
    "web_total_clicks", "web_cloud_count", "web_social_count", "web_job_count"
]

TARGET = "is_threat"
RANDOM_STATE = 42
SYNTHETIC_THREAT_ROWS = 5000


# =========================
# HELPER FUNCTION
# =========================

def evaluate_model(name, y_true, y_pred, results, show_cm=False):
    """Evaluate model performance using classification metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

    print(f"\n{name}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")

    # Display confusion matrix for main model
    if show_cm:
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Normal", "Threat"]
        )
        disp.plot()
        plt.title(f"{name} Confusion Matrix")
        plt.savefig(CONFUSION_MATRIX_PATH, bbox_inches="tight")
        plt.show()


# =========================
# MAIN PIPELINE
# =========================

def main():

    # =========================
    # LOAD DATA
    # =========================
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    # =========================
    # DATA CLEANING
    # =========================

    # Convert features to numeric
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    # Remove rows with missing values
    df = df.dropna(subset=FEATURES + [TARGET])
    df[TARGET] = df[TARGET].astype(int)

    print(f"Cleaned dataset size: {len(df)}")

    # =========================
    # TRAIN-TEST SPLIT
    # =========================

    # Split before augmentation to prevent data leakage
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        df[FEATURES],
        df[TARGET],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df[TARGET]
    )

    # =========================
    # CTGAN AUGMENTATION
    # =========================

    print("\nApplying CTGAN augmentation...")

    # Extract minority class (threat samples)
    train_threats = X_train_raw[y_train_raw == 1].copy()

    # Train CTGAN model
    metadata = Metadata.detect_from_dataframe(data=train_threats)
    synthesizer = CTGANSynthesizer(metadata)
    synthesizer.fit(train_threats)

    # Generate synthetic threat data
    synthetic_threats = synthesizer.sample(num_rows=SYNTHETIC_THREAT_ROWS)

    # Combine real and synthetic data
    X_train_final = pd.concat(
        [X_train_raw.reset_index(drop=True), synthetic_threats],
        ignore_index=True
    )

    y_train_final = pd.concat(
        [y_train_raw.reset_index(drop=True), pd.Series([1] * SYNTHETIC_THREAT_ROWS)],
        ignore_index=True
    )

    # =========================
    # FINAL SANITIZATION & SCALING
    # =========================

    # Ensure numeric format
    for col in FEATURES:
        X_train_final[col] = pd.to_numeric(X_train_final[col], errors="coerce")
        X_test_raw[col] = pd.to_numeric(X_test_raw[col], errors="coerce")

    # Remove any remaining invalid rows
    train_valid_index = X_train_final.dropna().index
    X_train_final = X_train_final.loc[train_valid_index].reset_index(drop=True)
    y_train_final = y_train_final.loc[train_valid_index].reset_index(drop=True)

    test_valid_index = X_test_raw.dropna().index
    X_test_final = X_test_raw.loc[test_valid_index].reset_index(drop=True)
    y_test_final = y_test_raw.loc[test_valid_index].reset_index(drop=True)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_final)
    X_test_scaled = scaler.transform(X_test_final)

    joblib.dump(scaler, SCALER_PATH)

    # Reshape for LSTM input (samples, timestep, features)
    X_train_3d = np.reshape(
        X_train_scaled,
        (X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
    )

    X_test_3d = np.reshape(
        X_test_scaled,
        (X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
    )

    # =========================
    # TRAIN LSTM MODEL
    # =========================

    print("\nTraining Sentinel LSTM model...")

    model = Sequential([
        Input(shape=(1, len(FEATURES))),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X_train_3d,
        y_train_final,
        epochs=20,
        batch_size=64,
        validation_data=(X_test_3d, y_test_final)
    )

    model.save(MODEL_PATH)

    # =========================
    # MODEL COMPARISON
    # =========================

    print("\nTraining comparison models...")
    results = []

    # Decision Tree
    dt_model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    dt_model.fit(X_train_scaled, y_train_final)
    evaluate_model("Decision Tree", y_test_final, dt_model.predict(X_test_scaled), results)

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf_model.fit(X_train_scaled, y_train_final)
    evaluate_model("Random Forest", y_test_final, rf_model.predict(X_test_scaled), results)

    # XGBoost
    xgb_model = xgb.XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE)
    xgb_model.fit(X_train_scaled, y_train_final)
    evaluate_model("XGBoost", y_test_final, xgb_model.predict(X_test_scaled), results)

    # SVM
    svm_model = LinearSVC(dual=False, max_iter=3000, random_state=RANDOM_STATE)
    svm_model.fit(X_train_scaled, y_train_final)
    evaluate_model("SVM (Linear)", y_test_final, svm_model.predict(X_test_scaled), results)

    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr_model.fit(X_train_scaled, y_train_final)
    evaluate_model("Logistic Regression", y_test_final, lr_model.predict(X_test_scaled), results)

    # LSTM Prediction
    lstm_probs = model.predict(X_test_3d)
    lstm_pred = (lstm_probs > 0.5).astype(int).flatten()
    evaluate_model("Sentinel LSTM", y_test_final, lstm_pred, results, show_cm=True)

    # Isolation Forest
    if_model = IsolationForest(contamination="auto", random_state=RANDOM_STATE)
    if_model.fit(X_train_scaled)
    if_preds = np.where(if_model.predict(X_test_scaled) == -1, 1, 0)
    evaluate_model("Isolation Forest", y_test_final, if_preds, results)

    # =========================
    # SAVE RESULTS
    # =========================

    results_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)
    results_df.to_csv(RESULTS_PATH, index=False)

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()
