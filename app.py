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

        bmi = weight / (height ** 2)

        data = np.array([[bmi]])
        data = scaler.transform(data)

        result = model.predict(data)

        if result[0] == 1:
            prediction = "You are Fit"
        else:
            prediction = "You are Not Fit"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
