import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np


st.set_page_config(
    page_title="Churn Prediction App",
    layout="wide",
    initial_sidebar_state="expanded"
)


model = joblib.load("models/churn_model.pkl")


st.markdown("""
<style>
    .main-title {
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 16px;
        color: #6b7280;
        margin-top: 0px;
    }
    .metric-card {
        padding: 18px;
        border-radius: 14px;
        background: white;
        border: 1px solid #e5e7eb;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.03);
        text-align: center;
    }
    .metric-label {
        font-size: 14px;
        color: #6b7280;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 800;
        color: #111827;
    }
    .section-header {
        font-size: 20px;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Helper Functions
# ----------------------------
def get_risk_level(prob):
    if prob < 0.35:
        return "Low Risk "
    elif prob < 0.65:
        return "Medium Risk "
    else:
        return "High Risk "


def plot_gauge(prob):
    fig, ax = plt.subplots(figsize=(6, 1.8))
    ax.axis("off")

    ax.barh(0, 1, height=0.35)
    ax.barh(0, prob, height=0.35)

    ax.text(0, 0.45, "0%", fontsize=12)
    ax.text(0.95, 0.45, "100%", fontsize=12)
    ax.text(prob, -0.2, f"{prob*100:.1f}%", fontsize=14, fontweight="bold")

    ax.set_xlim(0, 1)
    return fig


def get_shap_force_plot(pipeline_model, input_df):
    preprocessor = pipeline_model.named_steps["preprocessor"]
    clf_model = pipeline_model.named_steps["model"]

    # Transform input
    transformed = preprocessor.transform(input_df)

    # Convert sparse to dense (needed for SHAP sometimes)
    if hasattr(transformed, "toarray"):
        transformed_dense = transformed.toarray()
    else:
        transformed_dense = transformed

    explainer = shap.TreeExplainer(clf_model)
    shap_values = explainer.shap_values(transformed_dense)

    # Force plot for single sample
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        transformed_dense[0],
        matplotlib=True
    )

    return force_plot


# ----------------------------
# Header
# ----------------------------
st.markdown('<div class="main-title"> Customer Churn Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Enter customer details → predict churn risk → explain prediction using SHAP</div>', unsafe_allow_html=True)
st.markdown("---")

# ----------------------------
# Layout: Sidebar Inputs
# ----------------------------
st.sidebar.header(" Customer Input")

with st.sidebar.form("customer_form"):
    credit_score = st.number_input("Credit Score", 300, 900, 600)
    geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", 18, 100, 35)
    tenure = st.number_input("Tenure (years)", 0, 10, 5)
    balance = st.number_input("Balance", 0.0, 300000.0, 50000.0)
    num_products = st.number_input("Number of Products", 1, 4, 1)
    has_cr_card = st.selectbox("Has Credit Card?", [0, 1])
    is_active = st.selectbox("Is Active Member?", [0, 1])
    salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

    submitted = st.form_submit_button(" Predict Churn")

# ----------------------------
# Main Result Area
# ----------------------------
if submitted:
    input_data = pd.DataFrame([{
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active,
        "EstimatedSalary": salary
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    risk = get_risk_level(probability)

    # --- KPI Metrics Row ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Churn Probability</div>
            <div class="metric-value">{probability:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Risk Level</div>
            <div class="metric-value">{risk}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Model Decision</div>
            <div class="metric-value">{'Churn' if prediction == 1 else 'Not Churn'}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    if prediction == 1:
        st.error(" Model predicts this customer WILL churn.")
    else:
        st.success(" Model predicts this customer will NOT churn.")

    # --- Tabs Section ---
    tab1, tab2, tab3 = st.tabs([" Probability", " Explainability (SHAP)", " Report Download"])

    # ----------------------------
    # TAB 1: Probability
    # ----------------------------
    with tab1:
        st.markdown('<div class="section-header">Churn Probability Gauge</div>', unsafe_allow_html=True)
        fig = plot_gauge(probability)
        st.pyplot(fig)

    # ----------------------------
    # TAB 2: SHAP Explainability
    # ----------------------------
    with tab2:
        st.markdown('<div class="section-header">SHAP Force Plot (Single Prediction Explanation)</div>', unsafe_allow_html=True)
        st.info("Red features push the prediction toward churn, blue features push away.")

        if st.button(" Generate SHAP Force Plot"):
            with st.spinner("Generating SHAP force plot..."):
                force_plot = get_shap_force_plot(model, input_data)
                st.pyplot(force_plot, bbox_inches="tight")

            st.success(" SHAP force plot generated!")

    # ----------------------------
    # TAB 3: Report Download
    # ----------------------------
    with tab3:
        st.markdown('<div class="section-header">Download Prediction Report</div>', unsafe_allow_html=True)

        report_df = input_data.copy()
        report_df["Churn Probability"] = probability
        report_df["Risk Level"] = risk
        report_df["Prediction"] = "Churn" if prediction == 1 else "Not Churn"

        st.dataframe(report_df, use_container_width=True)

        csv = report_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label=" Download CSV Report",
            data=csv,
            file_name="churn_prediction_report.csv",
            mime="text/csv"
        )

else:
    st.info(" Fill in customer details from the sidebar and click **Predict Churn**.")
