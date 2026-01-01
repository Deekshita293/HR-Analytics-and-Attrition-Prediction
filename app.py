from flask import Flask, render_template, request, session
import joblib
import pandas as pd
import os
from waitress import serve

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))            
PROJECT_ROOT = os.path.dirname(BASE_DIR)                         

TEMPLATE_DIR = os.path.join(PROJECT_ROOT, "frontend", "Templates")
STATIC_DIR   = os.path.join(PROJECT_ROOT, "frontend", "static")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.secret_key = "supersecretkey"  # ✅ Needed for session

# Load model and encoders
model_path = os.path.join(BASE_DIR, "attrition_model.pkl")
model = joblib.load(model_path)

le_jobrole = joblib.load(os.path.join(BASE_DIR, "le_jobrole.pkl"))
le_overtime = joblib.load(os.path.join(BASE_DIR, "le_overtime.pkl"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        data = {
            "age": [int(request.form["age"])],
            "jobrole": [request.form["jobrole"]],
            "monthlyincome": [float(request.form["monthlyincome"])],
            "overtime": [request.form["overtime"]],
            "jobsatisfaction": [int(request.form["jobsatisfaction"])],
            "yearsatcompany": [int(request.form["yearsatcompany"])],
        }

        input_df = pd.DataFrame(data)

        # Encode categorical fields
        input_df["jobrole"] = le_jobrole.transform(input_df["jobrole"])
        input_df["overtime"] = le_overtime.transform(input_df["overtime"])

        # Predict
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        result = "High Risk of Leaving" if pred == 1 else "Low Risk of Leaving"
        probability = round(proba * 100, 2)

        # ✅ Save result in session
        session["last_result"] = result
        session["last_probability"] = probability
        session["last_satisfaction"] = int(request.form["jobsatisfaction"])

        return render_template("predict.html", result=result, probability=probability)

    return render_template("predict.html")

@app.route("/dashboard")
def dashboard():
    # ✅ Read prediction from session
    result = session.get("last_result", "No prediction yet")
    probability = session.get("last_probability", 0)
    satisfaction = session.get("last_satisfaction", 0)

    stats = {
        "high_risk": 1 if result == "High Risk of Leaving" else 0,
        "low_risk": 1 if result == "Low Risk of Leaving" else 0,
        "satisfaction": satisfaction
    }
    return render_template("dashboard.html", stats=stats, result=result, probability=probability)

@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username == "admin" and password == "admin123":
            return render_template("predict.html")
        else:
            return "Invalid credentials, please try again."
    return render_template("signin.html")

@app.route("/signout")
def signout():
    session.clear()  # ✅ Clear session on signout
    return render_template("index.html")

if __name__ == "__main__":
    print("✅ Server running at: http://127.0.0.1:5000")
    serve(app, host="127.0.0.1", port=5000)
