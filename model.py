import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("data/loan_approval_dataset.csv")

print("========== DATASET LOADED ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ==========================================
# 2. REMOVE DUPLICATE RECORDS
# ==========================================

df = df.drop_duplicates()

print("\nAfter removing duplicates:")
print("Rows:", df.shape[0])


# ==========================================
# 3. REMOVE UNNECESSARY ID COLUMNS
# ==========================================

df = df.drop(columns=["Loan ID", "Customer ID"])


# ==========================================
# 4. ENCODE CATEGORICAL COLUMNS
# ==========================================

encoder = LabelEncoder()

categorical_columns = df.select_dtypes(include=["object"]).columns

for column in categorical_columns:
    df[column] = encoder.fit_transform(df[column].astype(str))


# ==========================================
# 5. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("Loan_Approved", axis=1)
y = df["Loan_Approved"]


# ==========================================
# 6. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n========== PREPROCESSING COMPLETE ==========")
print("Training records:", X_train.shape[0])
print("Testing records:", X_test.shape[0])
print("Number of features:", X_train.shape[1])

print("\nFeatures used:")
print(X.columns.tolist())

print("\n============================================")
print("PREPROCESSING SUCCESSFULLY COMPLETED")
print("============================================")
# ==========================================
# 7. CLASSIFICATION ALGORITHMS
# ==========================================

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# Create models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier()
}


# ==========================================
# 8. TRAIN AND EVALUATE MODELS
# ==========================================

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1
    ])


# ==========================================
# 9. RESULT TABLE
# ==========================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Algorithm",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score"
    ]
)

print("\n========== CLASSIFICATION RESULTS ==========")
print(results_df.to_string(index=False))

print("\n============================================")
print("ALL MODELS TRAINED SUCCESSFULLY")
print("============================================")
# ==========================================
# 10. CONFUSION MATRIX
# ==========================================

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Use Random Forest for confusion matrix
rf_model = models["Random Forest"]

y_pred_rf = rf_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred_rf)

print("\n========== RANDOM FOREST CONFUSION MATRIX ==========")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Approved", "Approved"],
    yticklabels=["Not Approved", "Approved"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest - Confusion Matrix")

plt.tight_layout()

plt.savefig("static/images/confusion_matrix.png")

plt.show()
# RANDOM FOREST FEATURE IMPORTANCE

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)

plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Random Forest - Feature Importance")
plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig("static/images/feature_importance.png")

plt.show()
print("\nConfusion matrix saved successfully!")
# ==========================================
# 11. DATA VISUALIZATION
# ==========================================

import os

os.makedirs("static/images", exist_ok=True)

# 1. Loan Approval Distribution
plt.figure(figsize=(6, 5))
sns.countplot(x=df["Loan_Approved"])
plt.title("Loan Approval Distribution")
plt.xlabel("Loan Approved")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.savefig("static/images/loan_approval_distribution.png")
plt.show()


# 2. Credit Score Distribution
plt.figure(figsize=(7, 5))
sns.histplot(df["Credit Score"], kde=True)
plt.title("Credit Score Distribution")
plt.xlabel("Credit Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("static/images/credit_score_distribution.png")
plt.show()


# 3. Annual Income Distribution
plt.figure(figsize=(7, 5))
sns.histplot(df["Annual Income"], kde=True)
plt.title("Annual Income Distribution")
plt.xlabel("Annual Income")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("static/images/annual_income_distribution.png")
plt.show()


# 4. Current Loan Amount Distribution
plt.figure(figsize=(7, 5))
sns.histplot(df["Current Loan Amount"], kde=True)
plt.title("Current Loan Amount Distribution")
plt.xlabel("Current Loan Amount")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("static/images/loan_amount_distribution.png")
plt.show()


# 5. Loan Approval vs Credit Score
plt.figure(figsize=(7, 5))
sns.boxplot(x=df["Loan_Approved"], y=df["Credit Score"])
plt.title("Loan Approval vs Credit Score")
plt.xlabel("Loan Approved")
plt.ylabel("Credit Score")
plt.tight_layout()
plt.savefig("static/images/approval_vs_credit_score.png")
plt.show()


print("\n============================================")
print("DATA VISUALIZATION COMPLETED SUCCESSFULLY")
print("============================================")
# MODEL ACCURACY COMPARISON

accuracy_values = []
model_names = []

for name, model in models.items():
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred) * 100

    model_names.append(name)
    accuracy_values.append(accuracy)

plt.figure(figsize=(9, 5))

plt.bar(model_names, accuracy_values)

plt.xlabel("Classification Algorithms")
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Comparison")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig("static/images/model_accuracy_comparison.png")

plt.show()