# Heart Disease Prediction System

**Machine Learning Mini Project** · Python · Pandas · Scikit-learn · XGBoost · PostgreSQL · Streamlit

This project predicts whether a patient is likely to have heart disease from 11 routine clinical measurements. It uses the UCI Heart Disease dataset (920 patients from four hospitals). It trains three models: Logistic Regression as the baseline, then Random Forest and XGBoost. The models are compared on accuracy, precision, recall, specificity, F1, ROC-AUC and the confusion matrix. The chosen model is served in a Streamlit web app that saves every prediction to PostgreSQL.

---

## Results (held-out test set, 184 patients)

| Model | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | Missed sick patients | False alarms |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.799 | 0.804 | 0.843 | 0.744 | 0.823 | 0.888 | 16 | 21 |
| **Random Forest (selected)** | 0.799 | 0.788 | **0.873** | 0.707 | 0.828 | 0.899 | **13** | 24 |
| XGBoost | **0.815** | **0.815** | 0.863 | **0.756** | **0.838** | **0.900** | 14 | **20** |

**Which model is used, and why?** All three models score within about two percentage points of each other. The app recommends **Random Forest**. It had the highest F1 score in 5-fold cross-validation on the training data (0.828), and it catches the most sick patients (recall 0.873, only 13 missed out of 102).

The model is chosen on cross-validation scores, not on the test set. That keeps the test set a fair final check. XGBoost scores slightly higher on the test set, which shows how close the models are. You can switch between all three models in the app.

**What matters most to the models:** chest pain type (especially "asymptomatic"), chest pain during exercise, ST depression, maximum heart rate, age and sex (see `reports/feature_importance.png`).

---

## Project structure

```
heart_disease_prediction/
├── app.py                  Streamlit web app
├── train.py                Trains, compares and saves all three models
├── requirements.txt
├── data/
│   └── heart_disease_uci.csv
├── src/
│   ├── config.py           Paths, column lists, settings
│   ├── data.py             Loading, missing-value check, cleaning
│   ├── preprocessing.py    Filling gaps, one-hot encoding, scaling
│   ├── models.py           The three models and their tuning settings
│   ├── evaluate.py         Metrics and charts
│   └── database.py         PostgreSQL (SQLAlchemy) with SQLite fallback
├── sql/schema.sql          PostgreSQL tables (for reference)
├── notebooks/
│   └── heart_disease_walkthrough.ipynb   Step-by-step version for demos / viva
├── models/                 Saved models + model_info.json   (created by train.py)
└── reports/                Charts and metric tables          (created by train.py)
```

---

## How to run it (Windows)

Open **Command Prompt** or **PowerShell** in the `heart_disease_prediction` folder.

### 1. Create a virtual environment and install the libraries

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the models

```bat
python train.py
```

This takes under a minute. It prints the missing-value report, the cleaning steps, each model's scores and the final comparison. It also writes the models to `models/` and the charts to `reports/`.

> Always run this once on your own computer, even though trained models are included. Saved models can fail to load if your scikit-learn or XGBoost version differs from the one used to train them.

### 3. Start the web app

```bat
streamlit run app.py
```

Your browser opens at `http://localhost:8501`. The app has four tabs:

- **Predict**: enter a patient's details and get their estimated chance of heart disease, with a low / moderate / high risk band and the result from all three models.
- **Model comparison**: the metrics table, confusion matrices, ROC curves and feature importance.
- **Prediction history**: every prediction saved in the database, downloadable as CSV.
- **About the data**: the dataset and the cleaning steps.

Without PostgreSQL set up, the app still works. It saves predictions to a local SQLite file (`heart_app.db`) instead.

---

## Connecting PostgreSQL

1. Install PostgreSQL (with pgAdmin) from <https://www.postgresql.org/download/windows/>.
2. Create the database, in pgAdmin or in `psql -U postgres`:
   ```sql
   CREATE DATABASE heart_disease;
   ```
3. Tell the project how to connect. Replace `yourpassword` with your postgres password.

   **PowerShell**
   ```powershell
   $env:DATABASE_URL = "postgresql+psycopg2://postgres:yourpassword@localhost:5432/heart_disease"
   ```
   **Command Prompt**
   ```bat
   set DATABASE_URL=postgresql+psycopg2://postgres:yourpassword@localhost:5432/heart_disease
   ```
   This lasts only for the current window. Set it again in every new window, or add it under Windows *Environment Variables*.
4. Load the dataset into PostgreSQL and train:
   ```bat
   python train.py --to-db
   ```
   This creates the `heart_disease_raw` table (all 920 rows) and the `predictions` table.
5. Optional: train straight from the database instead of the CSV:
   ```bat
   python train.py --from-db
   ```
6. Run `streamlit run app.py`. The sidebar should say **Database: ✅ PostgreSQL**.

Useful queries:

```sql
SELECT dataset, COUNT(*), ROUND(AVG((num > 0)::int), 2) AS disease_rate
FROM heart_disease_raw GROUP BY dataset;

SELECT model_name, COUNT(*), ROUND(AVG(probability)::numeric, 2)
FROM predictions GROUP BY model_name;
```

---

## Method in brief

1. **Missing values.** The data is checked column by column (`reports/missing_values.png`). Cholesterol and blood pressure values of 0 are impossible, so they are treated as missing (172 patients had cholesterol = 0). Two columns are more than half empty and are dropped: `ca` (66% missing) and `thal` (53% missing). The remaining gaps are filled inside the model pipeline, using the median for numbers and the most common value for categories.
2. **Hospital name not used.** The `dataset` column says which hospital a patient came from. It is left out on purpose. Disease rates differ a lot between hospitals (Switzerland has over 90% positive cases), so the model would learn the hospital instead of the patient.
3. **Target.** The original diagnosis runs from 0 to 4. It is turned into 0 = no heart disease and 1 = heart disease (508 vs 410 patients, a fairly balanced split).
4. **Encoding and scaling.** The six categorical inputs are one-hot encoded and the five numeric inputs are standardised. Both steps sit inside a scikit-learn `Pipeline`, so they are learnt from the training data only (no data leakage).
5. **Training.** The data is split 80/20 with stratification. Each model is tuned with `GridSearchCV` and 5-fold stratified cross-validation, optimising ROC-AUC.
6. **Evaluation.** The models are compared on accuracy, precision, recall, specificity, F1, ROC-AUC and confusion matrices. Recall gets the most attention, because a missed sick patient (false negative) is the most harmful error.

### Inputs used by the model

| Input | Meaning |
|---|---|
| age | Age in years |
| sex | Male / Female |
| cp | Chest pain type: typical angina, atypical angina, non-anginal, asymptomatic |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl (yes/no) |
| restecg | Resting ECG: normal, ST-T abnormality, LV hypertrophy |
| thalch | Maximum heart rate achieved |
| exang | Chest pain brought on by exercise (yes/no) |
| oldpeak | ST depression during exercise relative to rest |
| slope | Slope of the peak exercise ST segment: upsloping, flat, downsloping |

---

## References

- UCI Machine Learning Repository, Heart Disease dataset: Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988). <https://archive.ics.uci.edu/dataset/45/heart+disease>
- Scikit-learn, Supervised learning (Logistic Regression, Random Forest): <https://scikit-learn.org/stable/supervised_learning.html>
- XGBoost documentation: <https://xgboost.readthedocs.io>
- Streamlit documentation: <https://docs.streamlit.io>

> **Disclaimer:** this is a student project for learning purposes. It is not a medical device and must not be used for real diagnosis.
