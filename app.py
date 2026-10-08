from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

# Load saved model and feature columns
model = joblib.load(os.path.join(BASE_DIR, "1_app", "random_forest_model.pkl"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "1_app", "feature_columns.pkl"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = {
        "Administrative": int(request.form["Administrative"]),
        "Administrative_Duration": float(request.form["Administrative_Duration"]),
        "Informational": int(request.form["Informational"]),
        "Informational_Duration": float(request.form["Informational_Duration"]),
        "ProductRelated": int(request.form["ProductRelated"]),
        "ProductRelated_Duration": float(request.form["ProductRelated_Duration"]),
        "BounceRates": float(request.form["BounceRates"]),
        "ExitRates": float(request.form["ExitRates"]),
        "PageValues": float(request.form["PageValues"]),
        "SpecialDay": float(request.form["SpecialDay"]),
        "Month": request.form["Month"],
        "OperatingSystems": int(request.form["OperatingSystems"]),
        "Browser": int(request.form["Browser"]),
        "Region": int(request.form["Region"]),
        "TrafficType": int(request.form["TrafficType"]),
        "VisitorType": request.form["VisitorType"],
        "Weekend": request.form["Weekend"] == "True"
    }

    input_df = pd.DataFrame([data])

    # Create the same dummy columns used during training
    for column in feature_columns:

        if column.startswith("Month_"):
            month_name = column.replace("Month_", "")
            input_df[column] = (
                input_df["Month"] == month_name
            ).astype(int)

        elif column.startswith("VisitorType_"):
            visitor_type = column.replace("VisitorType_", "")
            input_df[column] = (
                input_df["VisitorType"] == visitor_type
            ).astype(int)

    # Remove original text columns
    input_df = input_df.drop(
        columns=["Month", "VisitorType"]
    )

    # Arrange columns exactly like training data
    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Prediction
    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    if prediction:
        result = "Customer is likely to make a Purchase."
    else:
        result = "Customer is likely NOT to make a Purchase."

    return render_template(
        "index.html",
        prediction=result,
        probability=round(probability * 100, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)