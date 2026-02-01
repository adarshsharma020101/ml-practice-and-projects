from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("bmi_model.pkl")
scaler = joblib.load("scaler.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""

    if request.method == "POST":

        height = float(request.form["height"])
        weight = float(request.form["weight"])

        # Calculate BMI
        bmi = weight / (height ** 2)

        # Create BMI category (same as training)
        if bmi < 18.5:
            bmi_cat = 0
        elif bmi < 25:
            bmi_cat = 1
        elif bmi < 30:
            bmi_cat = 2
        else:
            bmi_cat = 3

        # IMPORTANT: 2 FEATURES
        data = np.array([[bmi, bmi_cat]])

        # Scale
        data = scaler.transform(data)

        # Predict
        result = model.predict(data)

        if result[0] == 1:
            prediction = "You are Fit"
        else:
            prediction = "You are Not Fit"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)

    app.run(debug=True)
