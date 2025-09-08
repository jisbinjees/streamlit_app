import streamlit as st
import pandas as pd
import polars as pl
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np


# Add background image

def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("https://i.redd.it/nflafw75q6a91.jpg")


# Helper functions

def load_csv(file, use_pandas=False):
    try:
        if use_pandas:
            return pd.read_csv(file), "pandas"
        else:
            return pl.read_csv(file), "polars"
    except Exception:
        return pd.read_csv(file), "pandas"

def impute_column(df, col, dtype):
    if np.issubdtype(dtype, np.integer):
        df[col] = df[col].fillna(int(df[col].mean() if df[col].mean() is not None else 0))
    elif np.issubdtype(dtype, np.floating):
        df[col] = df[col].fillna(float(df[col].median() if df[col].median() is not None else 0.0))
    else:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "unknown")
    return df

def clean_data(df, drop_na=False, dedup=False, normalize=False, impute=False):
    if drop_na:
        df = df.dropna()
    if dedup:
        df = df.drop_duplicates()
    if normalize:
        num_cols = df.select_dtypes(include="number").columns
        df[num_cols] = (df[num_cols] - df[num_cols].mean()) / df[num_cols].std()
    if impute:
        for col in df.columns:
            df = impute_column(df, col, df[col].dtype)
    return df

def make_profile(df):
    try:
        profile = ProfileReport(df, explorative=True)
        return profile.to_html()
    except Exception as e:
        return f"<h3>⚠ Profiling failed: {e}</h3>" + df.describe(include="all").transpose().to_html()

def train_simple_model(df, target_col):
    df = df.dropna(subset=[target_col])
    X = df.drop(columns=[target_col])
    y = df[target_col]

    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "unknown")
        else:
            X[col] = X[col].replace([np.inf, -np.inf], np.nan).fillna(X[col].median())

    encoders = {}
    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le

    target_encoder = None
    if y.dtype == "object":
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y.astype(str))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if len(set(y)) < 20:
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        task = "classification"
    else:
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        task = "regression"

    return model, task, score, encoders, target_encoder, list(X.columns)


# Streamlit UI

st.title("📊 Data-Cleansing, Profiling & ML Tool")
st.write("Upload CSV → Clean → Profile → Train ML → Predict")

# Upload CSV
uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    df, kind = load_csv(uploaded, use_pandas=True)
    st.success(f"Loaded with {kind.upper()}")
    
    st.write("### Data Preview")
    st.dataframe(df.head())

    # Cleaning options
    st.write("### Cleaning Options")
    drop_na = st.checkbox("Drop missing values")
    dedup = st.checkbox("Remove duplicates")
    normalize = st.checkbox("Normalize numeric columns")
    impute = st.checkbox("Impute missing values")

    if st.button("Apply Cleaning"):
        df_cleaned = clean_data(df, drop_na, dedup, normalize, impute)
        st.dataframe(df_cleaned.head())
        st.session_state.cleaned_df = df_cleaned

    # Profiling
    if st.button("Generate Profiling Report"):
        html_report = make_profile(df)
        st.components.v1.html(html_report, height=700, scrolling=True)

    # Train model
    st.write("### Train Model")
    target_col = st.selectbox("Select target column", df.columns)
    if st.button("Train Model"):
        model, task, score, encoders, target_encoder, feature_cols = train_simple_model(df, target_col)
        st.session_state.model = model
        st.session_state.encoders = encoders
        st.session_state.target_encoder = target_encoder
        st.session_state.feature_cols = feature_cols
        st.session_state.task = task  
        st.success(f"Trained {task} model with score: {score:.3f}")

    # Prediction
    if "model" in st.session_state:
        st.write("### Make Prediction")
        user_input = {}
        for col in st.session_state.feature_cols:
            if df[col].dtype == "object":
                user_input[col] = st.selectbox(f"{col}", df[col].dropna().unique())
            else:
                user_input[col] = st.number_input(f"{col}", value=float(df[col].median()))
        if st.button("Predict"):
            input_df = pd.DataFrame([user_input])
            for col, le in st.session_state.encoders.items():
                input_df[col] = le.transform(input_df[col].astype(str))
            prediction = st.session_state.model.predict(input_df)[0]
            if st.session_state.task == "classification" and st.session_state.target_encoder:
                prediction = st.session_state.target_encoder.inverse_transform([prediction])[0]
            st.success(f"Prediction: {prediction}")
