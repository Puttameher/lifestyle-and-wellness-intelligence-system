# app.py
# Beginner-friendly Flask web UI - updated to fix metrics and recommendations

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
import json
import os
import io, base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# -------------------------
# Load saved files (models, encoders, metrics)
# -------------------------
print("Loading saved files...")

# scaler used for main models
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# encoders
with open("labelenc_gender.pkl", "rb") as f:
    le_gender = pickle.load(f)
with open("labelenc_risk.pkl", "rb") as f:
    le_risk = pickle.load(f)

# best models saved earlier by training script
with open("model_classification.pkl", "rb") as f:
    best_clf = pickle.load(f)
with open("model_regression.pkl", "rb") as f:
    best_reg = pickle.load(f)

# optional stress regressor (may not exist)
stress_model = None
stress_scaler = None
if os.path.exists("reg_stress_linear.pkl"):
    with open("reg_stress_linear.pkl","rb") as f:
        stress_model = pickle.load(f)
    if os.path.exists("scaler_for_stress.pkl"):
        with open("scaler_for_stress.pkl","rb") as f:
            stress_scaler = pickle.load(f)
    print("Loaded stress regressor.")

# load all classifiers/regressors (so we can show each model output)
clf_models = {}
for fname in os.listdir():
    if fname.startswith("clf_") and fname.endswith(".pkl"):
        key = fname.replace("clf_","").replace(".pkl","")
        with open(fname,"rb") as f:
            clf_models[key] = pickle.load(f)

reg_models = {}
for fname in os.listdir():
    if fname.startswith("reg_") and fname.endswith(".pkl"):
        key = fname.replace("reg_","").replace(".pkl","")
        with open(fname,"rb") as f:
            reg_models[key] = pickle.load(f)

# -------------------------
# Load and normalize metrics JSON (flexible)
# -------------------------
metrics_raw = {}
metrics = {"classifiers": {}, "regressors": {}, "best_classifier": None, "best_regressor": None}

if os.path.exists("model_metrics.json"):
    with open("model_metrics.json", "r") as f:
        try:
            metrics_raw = json.load(f)
        except Exception:
            metrics_raw = {}

# copy best names if present
metrics["best_classifier"] = metrics_raw.get("best_classifier", metrics["best_classifier"])
metrics["best_regressor"] = metrics_raw.get("best_regressor", metrics["best_regressor"])

# Normalize many possible shapes into metrics["classifiers"][name]["accuracy"]
for key, val in metrics_raw.items():
    if key in ("best_classifier", "best_regressor"):
        continue

    # case: already nested dict {"classifiers": {...}}
    if key == "classifiers" and isinstance(val, dict):
        # ensure accuracy field is float
        for mname, mval in val.items():
            if isinstance(mval, dict) and "accuracy" in mval:
                metrics["classifiers"].setdefault(mname, {})["accuracy"] = float(mval["accuracy"])
            elif isinstance(mval, (int, float)):
                metrics["classifiers"].setdefault(mname, {})["accuracy"] = float(mval)
        continue

    if key == "regressors" and isinstance(val, dict):
        for mname, mval in val.items():
            if isinstance(mval, dict) and "r2" in mval:
                metrics["regressors"].setdefault(mname, {})["r2"] = float(mval["r2"])
            elif isinstance(mval, (int, float)):
                metrics["regressors"].setdefault(mname, {})["r2"] = float(mval)
        continue

    # flat numeric: e.g. {"knn": 0.8} => classifier accuracy
    if isinstance(val, (int, float)):
        if key.endswith("_r2"):
            regname = key[:-3]
            metrics["regressors"].setdefault(regname, {})["r2"] = float(val)
        else:
            metrics["classifiers"].setdefault(key, {})["accuracy"] = float(val)
        continue

    # flat dict per-model: e.g. {"knn": {"accuracy":0.8}} or {"linear_regression_r2": 0.5}
    if isinstance(val, dict):
        # try to find numeric fields
        if "accuracy" in val:
            metrics["classifiers"].setdefault(key, {})["accuracy"] = float(val["accuracy"])
            continue
        if "r2" in val:
            metrics["regressors"].setdefault(key, {})["r2"] = float(val["r2"])
            continue
        # else, check nested numbers
        for subk, subv in val.items():
            if subk == "accuracy":
                metrics["classifiers"].setdefault(key, {})["accuracy"] = float(subv)
            if subk == "r2":
                metrics["regressors"].setdefault(key, {})["r2"] = float(subv)

print("Loaded metrics. Classifiers:", list(metrics["classifiers"].keys()), "Regressors:", list(metrics["regressors"].keys()))

# -------------------------
# Helpers for charts: convert matplotlib figure into base64 to embed in HTML
# -------------------------
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64

# -------------------------
# Build charts from dataset (so charts are informative and not just for the single input)
# -------------------------
# -------------------------
# Build charts from dataset (so charts are informative and not just for the single input)
# -------------------------
def build_charts():
    charts = {}
    
    # 1. Load Test Data for "Actual vs Predicted" and Comparison
    if os.path.exists("test_data_viz.pkl"):
        with open("test_data_viz.pkl", "rb") as f:
            test_data = pickle.load(f)
            
        # 1. CORRELATION HEATMAP
        # -------------------------------------------------------------------
        corr_dict = test_data.get("correlation_matrix", {})
        
        if corr_dict:
            # Reconstruct DataFrame from dict
            corr_df = pd.DataFrame.from_dict(corr_dict)
            
            fig = plt.figure(figsize=(10, 8))
            plt.style.use('dark_background')
            
            # Heatmap
            # Use a diverging palette (coolwarm) and center at 0
            sns.heatmap(corr_df, annot=True, fmt=".2f", cmap='coolwarm', center=0, 
                        square=True, linewidths=0.5, linecolor='black',
                        cbar_kws={"shrink": 0.8})
            
            # Text styling - Neon Cyan
            cyan_color = '#00f3ff'
            plt.title("Feature Correlation Matrix", color=cyan_color, fontsize=14, fontweight='bold', pad=20)
            plt.xticks(rotation=45, ha='right', color=cyan_color)
            plt.yticks(rotation=0, color=cyan_color)
            plt.tick_params(colors=cyan_color)
            
            # Colorbar text check
            cbar = fig.axes[-1]
            cbar.tick_params(colors=cyan_color)
            
            # Improve layout for rotated xticks
            plt.tight_layout()
            
            charts["correlation_heatmap"] = fig_to_base64(fig)
        else:
            charts["correlation_heatmap"] = None
            
        charts["model_charts"] = [] # Clear old individual charts list

        
        # -- Chart C: Classifier Accuracy Comparison --
        acc_dict = test_data.get("clf_accuracies", {})
        if acc_dict:
            fig = plt.figure(figsize=(8, 4))
            plt.style.use('dark_background')
            models = list(acc_dict.keys())
            scores = list(acc_dict.values())
            
            # Bar chart
            bars = plt.bar(models, scores, color=['#00f3ff', '#ff0055', '#ccff00', '#aa00ff'])
            plt.ylim(0, 1.1)
            
            # Text styling - Neon Cyan
            cyan_color = '#00f3ff'
            plt.title("Model Accuracy Comparison", color=cyan_color, fontsize=12, fontweight='bold')
            plt.ylabel("Accuracy", color=cyan_color)
            plt.tick_params(colors=cyan_color)
            
            # Add text labels (keep these white or maybe cyan? User asked for text of heatmap/accuracy to be blue. Data labels might be better white for contrast on bars, but let's stick to request or keep legible)
            # Let's simple make axis/titles blue as requested. Data labels inside/on bars often need contrast against bar color.
            # Bar colors are various. White is safest for data labels. 
            # User said "text of heatmap and accuracy barchart".
            # I will change axes and titles. I'll make the data labels Cyan too if they are outside, or White if inside. 
            # Previous code put them on top. Cyan on dark bg is fine.
            
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.2f}',
                         ha='center', va='bottom', color=cyan_color, fontweight='bold')
            
            charts["clf_accuracy_bar"] = fig_to_base64(fig)
            
    else:
        charts["model_charts"] = []
        charts["clf_accuracy_bar"] = None

    if os.path.exists("lifestyle_biometrics_dataset_2500.csv"):
        df = pd.read_csv("lifestyle_biometrics_dataset_2500.csv")

        # Stress vs Sleep scatter
        fig = plt.figure(figsize=(6,5))
        plt.style.use('dark_background')
        sns.scatterplot(data=df, x="sleep_duration", y="stress_level", alpha=0.6, color="#ffcc00")
        plt.title("Dataset: Stress vs Sleep", color="white")
        plt.grid(alpha=0.2)
        charts["stress_sleep"] = fig_to_base64(fig)

        # BMI vs Risk boxplot
        if "health_risk_level" in df.columns:
            fig = plt.figure(figsize=(6,5))
            plt.style.use('dark_background')
            sns.boxplot(data=df, x="health_risk_level", y="bmi",  palette="cool")
            plt.title("Dataset: BMI vs Risk", color="white")
            plt.grid(alpha=0.2)
            charts["bmi_risk"] = fig_to_base64(fig)
        else:
            charts["bmi_risk"] = None

        # Physical activity pie
        fig = plt.figure(figsize=(4,4))
        plt.style.use('dark_background')
        activity_counts = df["physical_activity"].value_counts().sort_index()
        # Custom neon colors
        colors = ['#00f3ff', '#ff0055', '#ccff00', '#aa00ff', '#00ff99']
        plt.pie(activity_counts, labels=activity_counts.index, autopct="%1.0f%%", colors=colors[:len(activity_counts)], 
                textprops={'color':"white"})
        plt.title("Dataset: Physical Activity", color="white")
        charts["lifestyle_pie"] = fig_to_base64(fig)
    else:
        charts["stress_sleep"] = None 
        charts["bmi_risk"] = None
        charts["lifestyle_pie"] = None
        
    return charts

charts_cache = build_charts()

# -------------------------
# Prediction helpers
# -------------------------
def predict_all(x_array, user_input_stress):
    x_scaled = scaler.transform(x_array)

    # classifier outputs
    clf_out = {}
    for name, model in clf_models.items():
        try:
            pred_enc = int(model.predict(x_scaled)[0])
            pred_label = le_risk.inverse_transform([pred_enc])[0]
        except Exception:
            pred_label = "Error"
        acc = None
        if metrics.get("classifiers") and metrics["classifiers"].get(name):
            acc = metrics["classifiers"][name].get("accuracy")
        clf_out[name] = {"prediction": pred_label, "accuracy": acc}

    # regressors
    reg_out = {}
    for name, model in reg_models.items():
        try:
            val = float(model.predict(x_scaled)[0])
        except Exception:
            val = None
        r2 = None
        if metrics.get("regressors") and metrics["regressors"].get(name):
            r2 = metrics["regressors"][name].get("r2")
        reg_out[name] = {"value": round(val,1) if val is not None else None, "r2": r2}

    # best models
    try:
        best_clf_enc = int(best_clf.predict(x_scaled)[0])
        best_clf_label = le_risk.inverse_transform([best_clf_enc])[0]
    except Exception:
        best_clf_label = "Error"
    try:
        best_reg_val = round(float(best_reg.predict(x_scaled)[0]),1)
    except Exception:
        best_reg_val = None

    # predicted stress
    if stress_model is not None:
        if stress_scaler is not None:
            x_s = stress_scaler.transform(x_array)
            s_val = float(stress_model.predict(x_s)[0])
        else:
            s_val = float(stress_model.predict(x_scaled)[0])
        predicted_stress = round(s_val,1)
    else:
        predicted_stress = float(user_input_stress)

    return clf_out, reg_out, {"name": metrics.get("best_classifier"), "label": best_clf_label}, {"name": metrics.get("best_regressor"), "value": best_reg_val}, predicted_stress

# -------------------------
# Routes
# -------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # read and clean inputs
        age = float(request.form.get("age"))
        gender = request.form.get("gender")
        height = float(request.form.get("height"))
        weight = float(request.form.get("weight"))
        bmi = round(weight / ((height/100.0)**2), 1)

        sleep_duration = float(request.form.get("sleep_duration"))
        
        # Calculate Sleep Quality based on lifestyle inputs (auto-calculated)
        # Base: 3
        # > 7.5 hrs: +1
        # < 6 hrs: -1
        # Stress > 7: -1
        # Activity > 3: +1
        # Alcohol > 0: -1
        calc_qual = 3.0
        if sleep_duration >= 7.5:
             calc_qual += 1
        elif sleep_duration < 6:
             calc_qual -= 1
        
        stress_level_input = float(request.form.get("stress_level"))
        if stress_level_input > 7:
             calc_qual -= 1
        
        physical_activity = int(request.form.get("physical_activity"))
        if physical_activity >= 4:
             calc_qual += 1
        
        alcohol = int(request.form.get("alcohol"))
        if alcohol > 0:
             calc_qual -= 1
             
        # Clamp to 1-5 range
        sleep_quality = int(max(1, min(5, round(calc_qual))))
        
        smoking = int(request.form.get("smoking"))
        rest_hr = float(request.form.get("rest_hr"))

        # encode gender
        gender_val = int(le_gender.transform([gender])[0])

        # input array
        x = np.array([[age, gender_val, height, weight, bmi,
                       sleep_duration, sleep_quality, physical_activity,
                       int(stress_level_input), smoking, alcohol, rest_hr]])

        # predictions
        clf_out, reg_out, best_clf, best_reg, pred_stress = predict_all(x, stress_level_input)

        # wellness score pick
        if "linear_regression" in reg_out and reg_out["linear_regression"]["value"] is not None:
            wellness_score_display = reg_out["linear_regression"]["value"]
        elif best_reg.get("value") is not None:
            wellness_score_display = best_reg["value"]
        else:
            wellness_score_display = None

        # insights (rules)
        insights = []
        if sleep_duration < 6:
            insights.append("Sleep is lower than recommended")
        if pred_stress >= 7:
            insights.append("Stress level is high")
        if bmi > 27:
            insights.append("BMI is slightly elevated")
        if rest_hr > 90:
            insights.append("Resting heart rate is elevated")

        # recommendations (conditional)
        recs = []
        if physical_activity < 3:
            recs.append("Increase daily physical activity (e.g., 30 min brisk walk).")
        if sleep_duration < 7 or sleep_quality < 3:
            recs.append("Improve sleep consistency — aim for 7–8 hours and regular bedtimes.")
            recs.append("Reduce caffeine after 6 PM and limit screen time before bed.")
        if pred_stress >= 7:
            recs.append("Practice stress reduction: short walks, breathing exercises, or 10-min meditation.")
        if smoking == 1:
            recs.append("Reduce or quit smoking — it greatly improves long-term wellness.")
        if alcohol >= 2:
            recs.append("Reduce alcohol consumption to improve sleep and wellness.")
        if bmi >= 25 and bmi < 30:
            recs.append("Consider a balanced diet to reduce BMI.")
        elif bmi >= 30:
            recs.append("High BMI detected — consider consulting a healthcare professional.")
        if rest_hr > 90:
            recs.append("Your resting heart rate is elevated — consider a medical check-up if persistent.")

        # prepare inputs for display
        inputs = {"age": int(age), "gender": gender, "height": int(height), "weight": int(weight),
                  "bmi": bmi, "sleep_duration": sleep_duration, "sleep_quality": sleep_quality,
                  "physical_activity": physical_activity, "stress_level_input": stress_level_input,
                  "smoking": smoking, "alcohol": alcohol, "rest_hr": rest_hr}

        charts = charts_cache

        return render_template("results.html",
                               inputs=inputs,
                               clf_outputs=clf_out,
                               reg_outputs=reg_out,
                               best_clf=best_clf,
                               best_reg=best_reg,
                               wellness_score=wellness_score_display,
                               predicted_stress=pred_stress,
                               insights=insights,
                               recs=recs,
                               charts=charts,
                               metrics=metrics)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
