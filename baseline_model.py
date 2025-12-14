# baseline_combined_model.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# -------------------------
# Load Morgan FP & ChemBERTa Features
# -------------------------
X_morgan = np.load('features.npy')               # shape (N, 2048)
X_chemberta = np.load('chemberta_features.npy')  # shape (N, 1536)
y = np.load('labels.npy')

print("Morgan Shape:", X_morgan.shape)
print("ChemBERTa Shape:", X_chemberta.shape)
print("Labels Shape:", y.shape)

# -------------------------
# Combine Features
# -------------------------
X = np.hstack((X_morgan, X_chemberta))  # (N, 3584)
print("\nFinal Combined Feature Shape:", X.shape)

# -------------------------
# Scale Features
# -------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------
# Train/Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# Train Random Forest Baseline
# -------------------------
rf_model = RandomForestClassifier(
    n_estimators=400,
    class_weight='balanced_subsample',
    n_jobs=-1,
    random_state=42
)

print("\n🔥 Training Random Forest (Morgan + ChemBERTa)...")
rf_model.fit(X_train, y_train)

# -------------------------
# Evaluate
# -------------------------
y_pred = rf_model.predict(X_test)

print(f"\n✅ Combined RF Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\n🔍 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Low', 'Moderate', 'High']))

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -------------------------
# Save Model
# -------------------------
joblib.dump(rf_model, 'rf_combined_model.pkl')
print("\n💾 Saved model as 'rf_combined_model.pkl'")

# Save scaler too
joblib.dump(scaler, 'scaler.pkl')
print("💾 Saved scaler as 'scaler.pkl'")
