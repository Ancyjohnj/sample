import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(
    os.path.join(BASE_DIR, "1_app", "random_forest_model.pkl")
)

feature_columns = joblib.load(
    os.path.join(BASE_DIR, "1_app", "feature_columns.pkl")
)

print("Model loaded successfully!")
print("Number of features:", len(feature_columns))