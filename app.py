"""
Heart Disease Prediction - Streamlit web app.

Run from the project folder (after `python train.py`):
    streamlit run app.py
"""
import json

import joblib
import pandas as pd
import streamlit as st

from src import config, database

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")

MODEL_INFO_PATH = config.MODELS_DIR / "model_info.json"


# ---------------------------------------------------------------------------
# Loading models and database (cached so it only happens once)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    models = {}
    for key in config.MODEL_DISPLAY_NAMES:
        path = config.MODELS_DIR / f"{key}.joblib"
        if path.exists():
            models[key] = joblib.load(path)
    return models


@st.cache_data
def load_model_info():
    return json.loads(MODEL_INFO_PATH.read_text()) if MODEL_INFO_PATH.exists() else None


@st.cache_resource
def connect_database():
    try:
        database.init_db()
        return True, database.backend_name()
    except Exception as exc:  # the app still works without a database
        return False, str(exc)


if not MODEL_INFO_PATH.exists():
    st.error("No trained models found. Run `python train.py` in the project folder first, "
             "then refresh this page.")
    st.stop()

try:
    models = load_models()
except Exception as exc:
    st.error("The saved models could not be loaded (usually a library version mismatch). "
             f"Run `python train.py` again to retrain them on this computer.\n\nDetails: {exc}")
    st.stop()

info = load_model_info()
db_ok, db_status = connect_database()
best_key = info["best_model"]


def risk_band(p: float):
    if p < 0.30:
        return "Low", "#2e7d32"
    if p < 0.60:
        return "Moderate", "#ef6c00"
    return "High", "#c62828"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("❤️ Heart Disease Prediction")
    st.caption("Machine Learning Mini Project")
    model_keys = list(models.keys())
    st.caption("★ = recommended model")
    chosen_key = st.selectbox(
        "Model used for prediction",
        model_keys,
        index=model_keys.index(best_key),
        format_func=lambda k: config.MODEL_DISPLAY_NAMES[k] + (" ★" if k == best_key else ""),
    )
    m = info["models"][chosen_key]["test_metrics"]
    st.markdown("**This model on unseen test patients**")
    c1, c2 = st.columns(2)
    c1.metric("Recall", f"{m['recall']:.0%}", help="Share of sick patients the model catches")
    c2.metric("Precision", f"{m['precision']:.0%}", help="Share of flagged patients who are really sick")
    c1.metric("F1 score", f"{m['f1']:.2f}")
    c2.metric("ROC-AUC", f"{m['roc_auc']:.2f}")
    st.divider()
    st.caption(f"Database: {'✅ ' + db_status if db_ok else '⚠️ not connected'}")
    st.caption(f"Models trained: {info['trained_at']}")

tab_predict, tab_compare, tab_history, tab_data = st.tabs(
    ["🩺 Predict", "📊 Model comparison", "🗂️ Prediction history", "📁 About the data"])

# ---------------------------------------------------------------------------
# Tab 1: Predict
# ---------------------------------------------------------------------------
with tab_predict:
    st.header("Enter patient information")
    st.caption("All fields use the same units as the UCI Heart Disease dataset.")

    with st.form("patient_form"):
        patient_ref = st.text_input("Patient name or ID (optional)", max_chars=100)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Basic details")
            age = st.number_input("Age (years)", 18, 100, 54)
            sex = st.selectbox("Sex", config.CATEGORY_OPTIONS["sex"])
            trestbps = st.number_input("Resting blood pressure (mm Hg)", 80, 220, 130)
            chol = st.number_input("Serum cholesterol (mg/dl)", 100, 600, 230)
        with col2:
            st.subheader("Symptoms & tests")
            cp = st.selectbox(
                "Chest pain type", config.CATEGORY_OPTIONS["cp"],
                help="Typical angina: classic heart-related chest pain. Asymptomatic: no chest pain.")
            fbs = st.radio("Fasting blood sugar above 120 mg/dl?",
                           config.CATEGORY_OPTIONS["fbs"], horizontal=True)
            restecg = st.selectbox("Resting ECG result", config.CATEGORY_OPTIONS["restecg"])
        with col3:
            st.subheader("Exercise test")
            thalch = st.number_input("Maximum heart rate achieved", 60, 220, 140)
            exang = st.radio("Chest pain during exercise?",
                             config.CATEGORY_OPTIONS["exang"], horizontal=True)
            oldpeak = st.number_input("ST depression (oldpeak)", -3.0, 7.0, 0.5, step=0.1,
                                      help="Drop in the ECG ST segment during exercise compared with rest.")
            slope = st.selectbox("Slope of peak exercise ST segment", config.CATEGORY_OPTIONS["slope"])

        submitted = st.form_submit_button("Predict", type="primary", width="stretch")

    if submitted:
        patient = {
            "age": int(age), "sex": sex, "cp": cp, "trestbps": float(trestbps),
            "chol": float(chol), "fbs": fbs, "restecg": restecg, "thalch": float(thalch),
            "exang": exang, "oldpeak": float(oldpeak), "slope": slope,
        }
        X_new = pd.DataFrame([patient])[config.ALL_FEATURES]
        prob = float(models[chosen_key].predict_proba(X_new)[0, 1])
        pred = int(prob >= info["decision_threshold"])
        band, color = risk_band(prob)

        st.divider()
        left, right = st.columns([1, 2])
        with left:
            st.markdown(
                f"<div style='padding:1.2rem;border-radius:12px;background:{color}15;"
                f"border:2px solid {color};text-align:center'>"
                f"<div style='font-size:0.95rem;color:{color}'>Estimated chance of heart disease</div>"
                f"<div style='font-size:3rem;font-weight:700;color:{color}'>{prob:.0%}</div>"
                f"<div style='font-size:1.1rem;font-weight:600;color:{color}'>{band} risk</div></div>",
                unsafe_allow_html=True)
        with right:
            if pred == 1:
                st.error("**Prediction: heart disease likely.** The model places this patient in the "
                         "group that had heart disease in the training data.")
            else:
                st.success("**Prediction: heart disease unlikely.** The model places this patient in the "
                           "group that did not have heart disease in the training data.")
            st.caption(f"Model: {config.MODEL_DISPLAY_NAMES[chosen_key]} · "
                       f"cut-off {info['decision_threshold']:.0%}")

            # Compare the same patient across all three models.
            others = {config.MODEL_DISPLAY_NAMES[k]: float(mdl.predict_proba(X_new)[0, 1])
                      for k, mdl in models.items()}
            st.markdown("**All three models for this patient**")
            st.dataframe(
                pd.DataFrame({"Model": others.keys(), "Chance of disease": others.values()}),
                hide_index=True, width="stretch",
                column_config={"Chance of disease": st.column_config.ProgressColumn(
                    format="%.2f", min_value=0.0, max_value=1.0)})

        st.info("⚠️ This tool is a student project for learning purposes. "
                "It is not a medical diagnosis. Please consult a doctor for any health concerns.")

        if db_ok:
            try:
                database.save_prediction(patient, chosen_key, prob, pred, patient_ref)
                st.toast("Prediction saved to the database")
            except Exception as exc:
                st.warning(f"Prediction could not be saved: {exc}")

# ---------------------------------------------------------------------------
# Tab 2: Model comparison
# ---------------------------------------------------------------------------
with tab_compare:
    st.header("How the three models compare")
    st.write(
        f"Each model was tuned with {config.CV_FOLDS}-fold cross-validation on "
        f"{info['n_train']} training patients and then checked on {info['n_test']} patients it "
        f"had never seen. The recommended model is **{config.MODEL_DISPLAY_NAMES[best_key]}**, "
        "chosen by the highest cross-validated F1 score, which balances catching sick patients "
        "with avoiding false alarms.")

    rows = []
    for k, v in info["models"].items():
        t, c = v["test_metrics"], v["cv_metrics"]
        rows.append({
            "Model": config.MODEL_DISPLAY_NAMES[k],
            "Accuracy": t["accuracy"], "Precision": t["precision"], "Recall": t["recall"],
            "Specificity": t["specificity"], "F1": t["f1"], "ROC-AUC": t["roc_auc"],
            "Missed sick patients": t["false_negatives"], "False alarms": t["false_positives"],
            "CV F1": c["f1"], "CV Recall": c["recall"],
        })
    table = pd.DataFrame(rows).set_index("Model")
    st.dataframe(
        table.style.format({c: "{:.3f}" for c in table.columns
                            if c not in ("Missed sick patients", "False alarms")})
        .highlight_max(axis=0, color="#c8e6c9",
                       subset=["Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC-AUC", "CV F1", "CV Recall"])
        .highlight_min(axis=0, color="#c8e6c9", subset=["Missed sick patients", "False alarms"]),
        width="stretch")
    st.caption("Green = best in that column. All scores are on the held-out test set unless marked CV "
               "(cross-validation average on the training set).")

    with st.expander("What do these measures mean?"):
        st.markdown(
            "- **Accuracy**: share of all patients classified correctly.\n"
            "- **Precision**: of the patients flagged as sick, how many really were.\n"
            "- **Recall**: of the patients who really were sick, how many were caught. "
            "This matters most here, because missing a sick patient is the most harmful mistake.\n"
            "- **Specificity**: of the healthy patients, how many were correctly cleared.\n"
            "- **F1 score**: one number that balances precision and recall.\n"
            "- **ROC-AUC**: how well the model ranks sick patients above healthy ones (1.0 is perfect, 0.5 is guessing).")

    for title, fname in [("Confusion matrices", "confusion_matrices.png"),
                         ("All metrics side by side", "metric_comparison.png"),
                         ("ROC curves", "roc_curves.png"),
                         ("Most important inputs", "feature_importance.png")]:
        path = config.REPORTS_DIR / fname
        if path.exists():
            st.subheader(title)
            st.image(str(path), width="stretch")

    st.subheader("Best settings found for each model")
    st.json({config.MODEL_DISPLAY_NAMES[k]: v["best_params"] for k, v in info["models"].items()})

# ---------------------------------------------------------------------------
# Tab 3: Prediction history (from the database)
# ---------------------------------------------------------------------------
with tab_history:
    st.header("Saved predictions")
    if not db_ok:
        st.warning(f"Database not connected: {db_status}")
    else:
        history = database.fetch_predictions()
        if history.empty:
            st.info("No predictions yet. Make one in the Predict tab.")
        else:
            history["model_name"] = history["model_name"].map(config.MODEL_DISPLAY_NAMES).fillna(history["model_name"])
            history["result"] = history["prediction"].map({1: "Disease likely", 0: "Disease unlikely"})
            history["created_at"] = pd.to_datetime(history["created_at"], utc=True).dt.tz_convert(None)
            c1, c2, c3 = st.columns(3)
            c1.metric("Predictions saved", len(history))
            c2.metric("Flagged as disease likely", int(history["prediction"].sum()))
            c3.metric("Average chance of disease", f"{history['probability'].mean():.0%}")
            ordered = ["created_at", "patient_ref", "result", "probability", "model_name"] + config.ALL_FEATURES
            nice_names = {"created_at": "Time (UTC)", "patient_ref": "Patient", "result": "Result",
                          "model_name": "Model", **config.FEATURE_LABELS}
            st.dataframe(
                history[ordered].rename(columns=nice_names), hide_index=True, width="stretch",
                column_config={"probability": st.column_config.ProgressColumn(
                    "Chance of disease", format="%.2f", min_value=0.0, max_value=1.0)})
            st.download_button("Download as CSV", history.to_csv(index=False).encode(),
                               "predictions.csv", "text/csv")

# ---------------------------------------------------------------------------
# Tab 4: About the data
# ---------------------------------------------------------------------------
with tab_data:
    st.header("UCI Heart Disease dataset")
    st.write(
        f"The dataset has 920 patient records from four hospitals: Cleveland, Hungary, "
        f"Switzerland and Long Beach VA. After cleaning, {info['n_patients']} patients and "
        f"{len(config.ALL_FEATURES)} inputs were used. The original diagnosis (0 to 4) was turned into "
        "a yes/no target: 0 means no heart disease, 1 to 4 means heart disease is present.")
    st.markdown(
        "**Cleaning steps**\n"
        "1. Cholesterol and blood pressure values of 0 were treated as missing (they are impossible values).\n"
        "2. Columns with more than half their values missing were dropped: *ca* (66% missing) and *thal* (53% missing).\n"
        "3. The hospital name was not used as an input, so the model learns from the patient and not the hospital.\n"
        "4. Remaining gaps were filled in inside the model pipeline: numbers with the median, categories with the most common value.\n"
        "5. Categories were one-hot encoded and numbers were standardised.")
    c1, c2 = st.columns(2)
    for col, fname in [(c1, "missing_values.png"), (c2, "class_balance.png")]:
        path = config.REPORTS_DIR / fname
        if path.exists():
            col.image(str(path), width="stretch")
    st.caption("Source: UCI Machine Learning Repository – Heart Disease dataset "
               "(Janosi, Steinbrunn, Pfisterer & Detrano, 1988).")
