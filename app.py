import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# === Streamlit Page Settings ===
st.set_page_config(page_title="📧 Spam Email Classifier Dashboard", layout="wide")
st.title("📧 Spam Email Classifier")
st.markdown("Predict whether an email is **Spam** or **Not Spam** using Machine Learning.")

# === Model Selection & Load ===
model_paths = {
    "XGBoost": r"F:\coding\spam_cl\saved_models\xgboost_best_model.pkl",
    "Random Forest": r"F:\coding\spam_cl\saved_models\random_forest_best_model.pkl",
    "Logistic Regression": r"saved_models/logistic_regression_best_model.pkl",
    "Naive Bayes":r"F:\coding\spam_cl\saved_models\naive_bayes_best_model.pkl",
    "Gradient Boost":r"F:\coding\spam_cl\saved_models\gradient_boosting_best_model.pkl"
}

col1, col2, col3 = st.columns([1, 2, 2])

with col1:
    st.subheader("🧠 Select Model")
    selected_model = st.selectbox("Choose ML Model", list(model_paths.keys()))
    model = joblib.load(model_paths[selected_model])

    # Extract expected feature names
    try:
        expected_features = model.named_steps["preprocessor"].get_feature_names_out()
    except AttributeError:
        expected_features = model.feature_names_in_

    expected_features = list(expected_features)

with col2:
    st.subheader("🔧 Enter Features Manually")
    manual_input = {}

    # Get input for the first 10 important features
    for col in expected_features[:10]:
        manual_input[col] = st.slider(col, 0.0, 100.0, 0.0)

    # Set remaining features to 0
    for col in expected_features[10:]:
        manual_input[col] = 0.0

    df = pd.DataFrame([manual_input])[expected_features]

with col3:
    st.subheader("📊 Prediction Output")

    if st.button("🚀 Predict"):
        prediction = model.predict(df)
        proba = model.predict_proba(df)[:, 1][0]

        result = "🔴 SPAM" if prediction[0] == 1 else "🟢 NOT SPAM"
        spam_prob = round(proba * 100, 2)
        not_spam_prob = round(100 - spam_prob, 2)

        # Show Metrics
        st.metric("Result", result)
        st.metric("Spam Probability", f"{spam_prob}%")
        st.metric("Not Spam Probability", f"{not_spam_prob}%")

        # Pie chart
        st.subheader("📈 Probability Distribution")
        fig, ax = plt.subplots()
        ax.pie([not_spam_prob, spam_prob],
               labels=["Not Spam", "Spam"],
               autopct='%1.1f%%',
               colors=["#66c2a5", "#fc8d62"])
        st.pyplot(fig)
