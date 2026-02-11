from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

FEATURES = [
    ("Age", "num"),
    ("Number of sexual partners", "num"),
    ("First sexual intercourse", "num"),
    ("Num of pregnancies", "num"),
    ("Smokes", "bin"),
    ("Smokes (years)", "num"),
    ("Smokes (packs/year)", "num"),
    ("Hormonal Contraceptives", "bin"),
    ("Hormonal Contraceptives (years)", "num"),
    ("IUD", "bin"),
    ("IUD (years)", "num"),
    ("STDs", "bin"),
    ("STDs (number)", "num"),
    ("STDs:condylomatosis", "bin"),
    ("STDs:cervical condylomatosis", "bin"),
    ("STDs:vaginal condylomatosis", "bin"),
    ("STDs:vulvo-perineal condylomatosis", "bin"),
    ("STDs:syphilis", "bin"),
    ("STDs:pelvic inflammatory disease", "bin"),
    ("STDs:genital herpes", "bin"),
    ("STDs:molluscum contagiosum", "bin"),
    ("STDs:AIDS", "bin"),
    ("STDs:HIV", "bin"),
    ("STDs:Hepatitis B", "bin"),
    ("STDs:HPV", "bin"),
    ("STDs: Number of diagnosis", "num"),
    ("Dx:Cancer", "bin"),
    ("Dx:CIN", "bin"),
    ("Dx:HPV", "bin"),
    ("Dx", "bin"),
    ("Hinselmann", "bin"),
    ("Schiller", "bin"),
    ("Citology", "bin")
]

@app.route("/")
def home():
    return render_template("index.html", features=FEATURES, result=None)

@app.route("/predict", methods=["POST"])
def predict():
    data = []
    for i in range(len(FEATURES)):
        data.append(float(request.form[f"f{i}"]))

    data = np.array(data).reshape(1, -1)
    data = scaler.transform(data)
    pred = model.predict(data)[0]

    result = "🚨 Cervical Cancer Detected" if pred == 1 else "✅ No Cervical Cancer"
    return render_template("index.html", features=FEATURES, result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)


