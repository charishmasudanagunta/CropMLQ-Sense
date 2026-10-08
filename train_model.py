import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==============================
# 1. Load AGRIQ-SENSE dataset
# ==============================

DATA_FILE = "AGRIQ_SENSE_200_dataset.xlsx"

df = pd.read_excel(DATA_FILE)

print("\nDataset loaded successfully!")
print("Number of samples:", len(df))
print("Columns:", list(df.columns))


# ==============================
# 2. Features
# ==============================

features = [
    "Crop",
    "Temperature (°C)",
    "Humidity (%)",
    "Rainfall (mm)",
    "Soil Moisture (%)"
]

X = df[features]

# Disease is our main prediction target
y_disease = df["Disease"]

# Risk is our second prediction target
y_risk = df["Risk Level"]


# ==============================
# 3. Preprocessing
# ==============================

categorical_features = ["Crop"]

numeric_features = [
    "Temperature (°C)",
    "Humidity (%)",
    "Rainfall (mm)",
    "Soil Moisture (%)"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "crop",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# ==============================
# 4. Disease prediction model
# ==============================

disease_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_disease,
    test_size=0.20,
    random_state=42,
    stratify=y_disease
)


print("\nTraining disease model...")

disease_model.fit(X_train, y_train)

disease_predictions = disease_model.predict(X_test)

disease_accuracy = accuracy_score(
    y_test,
    disease_predictions
)

print("\nDisease Model Accuracy:")
print(f"{disease_accuracy * 100:.2f}%")

print("\nDisease Classification Report:")
print(
    classification_report(
        y_test,
        disease_predictions,
        zero_division=0
    )
)


# ==============================
# 5. Train risk model
# ==============================

risk_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)


X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X,
    y_risk,
    test_size=0.20,
    random_state=42,
    stratify=y_risk
)


print("\nTraining risk model...")

risk_model.fit(X_train_r, y_train_r)

risk_predictions = risk_model.predict(X_test_r)

risk_accuracy = accuracy_score(
    y_test_r,
    risk_predictions
)

print("\nRisk Model Accuracy:")
print(f"{risk_accuracy * 100:.2f}%")

print("\nRisk Classification Report:")
print(
    classification_report(
        y_test_r,
        risk_predictions,
        zero_division=0
    )
)


# ==============================
# 6. Save trained models
# ==============================

joblib.dump(
    disease_model,
    "disease_model.pkl"
)

joblib.dump(
    risk_model,
    "risk_model.pkl"
)


print("\n================================")
print("TRAINING COMPLETE")
print("================================")

print("Created:")
print("  disease_model.pkl")
print("  risk_model.pkl")

print("\nThese models can now be used by app.py.")