import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

# Load dataset
df = pd.read_csv("data/data_cleaned.csv")

# Target and dropped columns
target = "Dropout"
drop_cols = ["Student_ID", "Name", "Dept", "Year", "payment_status", "risk_score"]

X = df.drop(columns=drop_cols + [target])
y = df[target]

# Features left after dropping
numeric_features = ["Attendance%", "Avg_Marks", "Assignments_Submitted%", "Backlogs"]

# ---- 📊 Correlation Matrix ----
plt.figure(figsize=(8, 6))
sns.heatmap(X.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Features")
plt.show()

# Pipeline for numeric preprocessing
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Column transformer (only numeric now)
preprocessor = ColumnTransformer([
    ("num", num_pipe, numeric_features)
])

# Full pipeline with classifier
clf = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train model
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))

# ---- 📊 Confusion Matrix Plot ----
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ---- 📊 Feature Importance ----
rf = clf.named_steps["classifier"]
importances = rf.feature_importances_
feature_names = numeric_features

plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=feature_names, palette="viridis")
plt.title("Feature Importance")
plt.show()

# Save model
joblib.dump(clf, "model.joblib")
print("Model saved as model.joblib")

# Testing Accuracy
test_accuracy = accuracy_score(y_test, y_pred)
print("Testing Accuracy:", test_accuracy)

