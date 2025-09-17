# app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

# -------------------------
# Helpers: load model safely
# -------------------------
def try_joblib_load(paths):
    for p in paths:
        if Path(p).exists():
            try:
                obj = joblib.load(p)
                return obj, p
            except Exception as e:
                st.warning(f"Found file {p} but loading failed: {e}")
    return None, None

@st.cache_resource(show_spinner=False)
def load_models_and_preprocessor():
    # 1) Load models (try common names)
    model_dir = Path("models")
    model_paths = {
        "dt": [
            model_dir / "loan_approval_dt_model.pkl",
            model_dir / "loan_approval_dt_model",
            "loan_approval_dt_model.pkl",
            "loan_approval_dt_model"
        ],
        "reg": [
            model_dir / "loan_approval_reg_model.pkl",
            model_dir / "loan_approval_reg_model",
            "loan_approval_reg_model.pkl",
            "loan_approval_reg_model"
        ],
    }

    dt_model, dt_path = try_joblib_load([str(p) for p in model_paths["dt"]])
    reg_model, reg_path = try_joblib_load([str(p) for p in model_paths["reg"]])

    # 2) load preprocessor if user saved it earlier
    preprocessor_paths = [
        model_dir / "preprocessor.pkl",
        model_dir / "preprocessor.joblib",
        "preprocessor.pkl",
        "preprocessor.joblib"
    ]
    preprocessor, pp_path = try_joblib_load([str(p) for p in preprocessor_paths])

    # 3) If preprocessor not found, try to load training CSV and build the same preprocessor used in notebook
    df = None
    if preprocessor is None:
        candidate_csv = Path("loan_approval_dataset.csv")
        if candidate_csv.exists():
            df = pd.read_csv(candidate_csv)
            # replicate notebook cleaning steps
            df.columns = df.columns.str.strip()
            if 'loan_id' in df.columns:
                df = df.drop(columns=['loan_id'])
            # target is loan_status
            if 'loan_status' not in df.columns:
                raise RuntimeError("Dataset found but 'loan_status' column not present.")
            X = df.drop(columns=['loan_status'])
            # derive numeric / categorical columns same as notebook
            num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

            # build transformers identical to the notebook
            numeric_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            categorical_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, num_cols),
                    ("cat", categorical_transformer, cat_cols)
                ],
                remainder="drop"
            )
            # fit the preprocessor on the training features
            preprocessor.fit(X)
            pp_path = str(candidate_csv) + " (preprocessor fit from CSV)"
        else:
            # neither preprocessor nor CSV found
            preprocessor = None

    # Return everything and some metadata (if df loaded)
    meta = {}
    if df is not None:
        X = df.drop(columns=['loan_status'])
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        meta = {
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "cat_options": {c: sorted(X[c].dropna().unique().tolist()) for c in cat_cols},
            "num_medians": X[num_cols].median().to_dict(),
            "feature_order": X.columns.tolist(),
            "target_values": sorted(df['loan_status'].dropna().unique().tolist())
        }

    return {
        "dt_model": dt_model, "dt_path": dt_path,
        "reg_model": reg_model, "reg_path": reg_path,
        "preprocessor": preprocessor, "preprocessor_src": pp_path,
        "df_meta": meta
    }

# -------------------------
# Load artifacts
# -------------------------
artifacts = load_models_and_preprocessor()

# Inform user if anything missing
if artifacts["dt_model"] is None and artifacts["reg_model"] is None:
    st.error("No models found. Put your trained models in the `models/` folder with names like "
             "`loan_approval_dt_model.pkl` and `loan_approval_reg_model.pkl`.")
    st.stop()

if artifacts["preprocessor"] is None:
    st.error("No preprocessor found and dataset CSV not found. Place `preprocessor.pkl` in `models/` "
             "or place the training CSV `loan_approval_dataset.csv` in the app folder and restart.")
    st.stop()

# pull variables
dt_model = artifacts["dt_model"]
reg_model = artifacts["reg_model"]
preprocessor = artifacts["preprocessor"]
meta = artifacts["df_meta"]

st.title("🏦 Loan Approval Predictor")
st.caption("Model deployment using the same preprocessing pipeline from the notebook.")

# choose model
model_choice = st.radio("Choose model to use for prediction", ("Logistic Regression", "Decision Tree"))

# Feature inputs (automatic UI from meta)
st.header("Applicant features")
if not meta:
    st.warning("No dataset metadata available. Please ensure `loan_approval_dataset.csv` is present.")
    st.stop()

feature_order = meta["feature_order"]
num_cols = meta["num_cols"]
cat_cols = meta["cat_cols"]
cat_options = meta["cat_options"]
num_medians = meta["num_medians"]

# Build input dict
ui_values = {}
cols_per_row = 2
# numeric inputs
for col in feature_order:
    if col in num_cols:
        default = num_medians.get(col, 0.0)
        ui_values[col] = st.number_input(label=col, value=float(default), step=1.0)
    elif col in cat_cols:
        opts = cat_options.get(col, [])
        if len(opts) > 0:
            # convert options to strings for selectbox
            opts_str = [str(x) for x in opts]
            ui_values[col] = st.selectbox(label=col, options=opts_str)
        else:
            ui_values[col] = st.text_input(label=col, value="")

# Button to predict
st.markdown("---")
threshold = st.slider("Decision threshold for choosing positive/approved class (useful if you want to adjust)", 0.0, 1.0, 0.5)
if st.button("Predict"):
    # assemble input dataframe in the same column order used for preprocessing
    input_df = pd.DataFrame([{c: ui_values[c] for c in feature_order}])

    # Ensure correct dtypes for numeric columns
    for c in num_cols:
        # if the value is empty string convert to np.nan
        val = input_df.at[0, c]
        try:
            input_df[c] = pd.to_numeric(input_df[c])
        except Exception:
            input_df[c] = np.nan

    # Transform with preprocessor
    try:
        X_input = preprocessor.transform(input_df)
        # OneHotEncoder with sparse=False returns dense; but if sparse matrix returned convert
        if hasattr(X_input, "toarray"):
            X_input = X_input.toarray()
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        st.stop()

    # choose model
    model = reg_model if model_choice == "Logistic Regression" else dt_model

    # Predict
    try:
        raw_pred = model.predict(X_input)[0]
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        st.stop()

    # Attempt to get probabilities (some classifiers support predict_proba)
    proba = None
    try:
        proba = model.predict_proba(X_input)[0]
        classes = list(model.classes_)
    except Exception:
        proba = None
        classes = ["class_0", "class_1"]

    # Show results
    st.subheader("Result")
    st.write("Raw model output:", raw_pred)
    if proba is not None:
        prob_df = pd.DataFrame({"class": classes, "probability": proba})
        st.table(prob_df)

        # Decide positive class automatically:
        # If one of the classes looks like 'Y' or 'Approved' choose that as positive, else choose class[1]
        positive_class = None
        for c in classes:
            if str(c).lower() in ("y", "yes", "approved", "1", "true"):
                positive_class = c
                break
        if positive_class is None:
            # fallback to second class if >1
            positive_class = classes[1] if len(classes) > 1 else classes[0]

        pos_idx = classes.index(positive_class)
        pos_prob = proba[pos_idx]
        st.write(f"Selected positive class for decision: **{positive_class}** (probability = {pos_prob:.3f})")
        st.write("Decision at threshold {} : **{}**".format(threshold,
                 "APPROVED" if pos_prob >= threshold else "REJECTED"))
    else:
        st.write("Model does not provide probability estimates. Raw prediction:", raw_pred)

st.markdown("---")
st.write("Notes:")
st.write("- This app constructs the same preprocessing pipeline used during training (fitted from `loan_approval_dataset.csv`).")
st.write("- If you have previously saved the fitted preprocessor (`preprocessor.pkl`), place it inside `models/` to skip fitting from the CSV.")
st.write("- Make sure `models/loan_approval_dt_model.pkl` and `models/loan_approval_reg_model.pkl` exist (or similar file names).")
