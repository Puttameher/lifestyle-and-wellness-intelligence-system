# -----------------------------------------------------------
# 1. Import Libraries
# -----------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)

# -----------------------------------------------------------
# 2. Load Dataset
# -----------------------------------------------------------

df = pd.read_csv("lifestyle_biometrics_dataset_2500.csv")
print("\nFirst 5 rows:")
print(df.head())

# -----------------------------------------------------------
# 3. Dataset Info
# -----------------------------------------------------------

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------------------------------------
# 4. Basic Visualization (Optional)
# -----------------------------------------------------------

df.hist(figsize=(15, 10), bins=30)
plt.show()

# -----------------------------------------------------------
# 5. Label Encoding
# -----------------------------------------------------------

data = df.copy()
le_gender = LabelEncoder()
le_risk = LabelEncoder()

data["gender"] = le_gender.fit_transform(data["gender"])
data["health_risk_level"] = le_risk.fit_transform(data["health_risk_level"])

# -----------------------------------------------------------
# 6. Define Features and Targets
# -----------------------------------------------------------

features = [
    "age","gender","height","weight","bmi",
    "sleep_duration","sleep_quality","physical_activity",
    "stress_level","smoking","alcohol_consumption",
    "resting_heart_rate"
]

X = data[features]
y_class = data["health_risk_level"]
y_reg = data["wellness_score"]

# -----------------------------------------------------------
# 7. Train-Test Split
# -----------------------------------------------------------

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.2, random_state=0
)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_reg, test_size=0.2, random_state=0
)

# -----------------------------------------------------------
# 8. Scaling Features
# -----------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_c)
X_test_scaled = scaler.transform(X_test_c)

X_train_scaled_r = X_train_scaled
X_test_scaled_r = X_test_scaled

# -----------------------------------------------------------
# 9. CLASSIFICATION MODELS
# -----------------------------------------------------------

print("\n================ CLASSIFICATION MODELS ================\n")

# Logistic Regression
log_model = LogisticRegression(max_iter=500)
log_model.fit(X_train_scaled, y_train_c)
y_pred_log = log_model.predict(X_test_scaled)
acc_log = accuracy_score(y_test_c, y_pred_log)
print("Logistic Regression Accuracy:", acc_log)

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=200, random_state=0)
rf_model.fit(X_train_c, y_train_c)
y_pred_rf = rf_model.predict(X_test_c)
acc_rf = accuracy_score(y_test_c, y_pred_rf)
print("\nRandom Forest Accuracy:", acc_rf)

# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train_c)
y_pred_nb = nb_model.predict(X_test_scaled)
acc_nb = accuracy_score(y_test_c, y_pred_nb)
print("\nNaive Bayes Accuracy:", acc_nb)

# KNN Classifier
knn_model = KNeighborsClassifier(n_neighbors=7)
knn_model.fit(X_train_scaled, y_train_c)
y_pred_knn = knn_model.predict(X_test_scaled)
acc_knn = accuracy_score(y_test_c, y_pred_knn)
print("\nKNN Accuracy:", acc_knn)

# -----------------------------------------------------------
# 10. Compare Classification Accuracies
# -----------------------------------------------------------

print("\nModel Accuracy Comparison:")
print("Logistic Regression:", acc_log)
print("Random Forest:", acc_rf)
print("Naive Bayes:", acc_nb)
print("KNN:", acc_knn)

# Pick best classifier
best_clf_name = max(
    {
        "logistic_regression": acc_log,
        "random_forest": acc_rf,
        "naive_bayes": acc_nb,
        "knn": acc_knn
    },
    key=lambda k: {
        "logistic_regression": acc_log,
        "random_forest": acc_rf,
        "naive_bayes": acc_nb,
        "knn": acc_knn
    }[k]
)

print("\nBest Classifier:", best_clf_name)

# -----------------------------------------------------------
# 11. REGRESSION MODELS
# -----------------------------------------------------------

print("\n================= REGRESSION MODELS =================\n")

# Linear Regression
lr_reg = LinearRegression()
lr_reg.fit(X_train_scaled_r, y_train_r)
y_pred_lr_reg = lr_reg.predict(X_test_scaled_r)

r2_lin = r2_score(y_test_r, y_pred_lr_reg)
mae_lin = mean_absolute_error(y_test_r, y_pred_lr_reg)

print("Linear Regression R²:", r2_lin)
print("Linear Regression MAE:", mae_lin)

# Since Random Forest Regressor removed → Linear Regression is best
best_reg_name = "linear_regression"

# -----------------------------------------------------------
# 12. SAVE MODELS, SCALER, ENCODERS, METRICS (BEGINNER STYLE)
# -----------------------------------------------------------

print("\n================= SAVING MODELS =================")

# Save Scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save Label Encoders
with open("labelenc_gender.pkl", "wb") as f:
    pickle.dump(le_gender, f)

with open("labelenc_risk.pkl", "wb") as f:
    pickle.dump(le_risk, f)

# Save all individual models
with open("clf_logistic_regression.pkl", "wb") as f:
    pickle.dump(log_model, f)

with open("clf_random_forest.pkl", "wb") as f:
    pickle.dump(rf_model, f)

with open("clf_naive_bayes.pkl", "wb") as f:
    pickle.dump(nb_model, f)

with open("clf_knn.pkl", "wb") as f:
    pickle.dump(knn_model, f)

with open("reg_linear_regression.pkl", "wb") as f:
    pickle.dump(lr_reg, f)

# Save best models with fixed names
if best_clf_name == "logistic_regression":
    best_clf_model = log_model
elif best_clf_name == "random_forest":
    best_clf_model = rf_model
elif best_clf_name == "naive_bayes":
    best_clf_model = nb_model
else:
    best_clf_model = knn_model

with open("model_classification.pkl", "wb") as f:
    pickle.dump(best_clf_model, f)

with open("model_regression.pkl", "wb") as f:
    pickle.dump(lr_reg, f)

# Save metrics to JSON
metrics = {
    "logistic_regression": acc_log,
    "random_forest": acc_rf,
    "naive_bayes": acc_nb,
    "knn": acc_knn,
    "linear_regression_r2": r2_lin,
    "best_classifier": best_clf_name,
    "best_regressor": best_reg_name
}

with open("model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nAll models and metrics saved successfully!")

# -----------------------------------------------------------
# 13. SAVE TEST DATA FOR VISUALIZATION (APP)
# -----------------------------------------------------------

print("\nSaving test data for visualization...")

# Compute Correlation Matrix for the dataset
# Select numeric columns only
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr().to_dict()

test_data = {
    "y_test_reg": y_test_r.iloc[:50].tolist(), 
    "y_pred_reg": y_pred_lr_reg[:50].tolist(),
    "correlation_matrix": corr_matrix,
    "clf_accuracies": {
        "LogReg": acc_log,
        "RandForest": acc_rf,
        "NaiveBayes": acc_nb,
        "KNN": acc_knn
    },
    "reg_metrics": {
        "LinearReg R2": r2_lin
    }
}

with open("test_data_viz.pkl", "wb") as f:
    pickle.dump(test_data, f)
print("Test data saved to test_data_viz.pkl")

# -----------------------------------------------------------
# END OF PROJECT CODE
# -----------------------------------------------------------

print("\n🎉 Project Code Executed Successfully!")
