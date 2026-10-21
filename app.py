
import streamlit as st
import pandas as pd
import joblib
import os

# Page settings
st.set_page_config(
    page_title="Student Result AI Agent",
    page_icon="🎓",
    layout="wide"
)

# Load saved ML model
MODEL_PATH = "student_result_model.pkl"
DATA_PATH = "student-por.csv"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Check the project folder.")
    st.stop()

model = load_model()

# Header
st.title("🎓 Student Result Prediction AI Agent")
st.write(
    "Predict student results using Machine Learning "
    "and receive academic guidance."
)

st.divider()

# Student input form
st.subheader("Enter Student Details")

with st.form("student_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        g1 = st.number_input(
            "First Period Grade (G1)",
            min_value=0, max_value=20, value=10
        )

    with col2:
        g2 = st.number_input(
            "Second Period Grade (G2)",
            min_value=0, max_value=20, value=10
        )

    with col3:
        studytime = st.selectbox(
            "Weekly Study Time",
            options=[1, 2, 3, 4],
            index=1
        )

    col4, col5 = st.columns(2)

    with col4:
        failures = st.number_input(
            "Past Class Failures",
            min_value=0, max_value=3, value=0
        )

    with col5:
        absences = st.number_input(
            "Number of Absences",
            min_value=0, max_value=100, value=2
        )

    submitted = st.form_submit_button(
        "Predict Student Result",
        use_container_width=True
    )

# Prediction
if submitted:
    if not os.path.exists(DATA_PATH):
        st.error("Dataset file student-por.csv was not found.")
        st.stop()

    df = pd.read_csv(DATA_PATH, sep=";")

    # Use dataset structure for all required model features.
    # The five entered fields replace the sample values.
    student_data = df.drop(
        columns=["G3"],
        errors="ignore"
    ).iloc[[0]].copy()

    student_data.loc[:, "G1"] = g1
    student_data.loc[:, "G2"] = g2
    student_data.loc[:, "studytime"] = studytime
    student_data.loc[:, "failures"] = failures
    student_data.loc[:, "absences"] = absences

    try:
        prediction = model.predict(student_data)[0]
        probabilities = model.predict_proba(student_data)[0]
        classes = model.named_steps["classifier"].classes_

        probability_map = dict(zip(classes, probabilities))
        pass_prob = float(probability_map.get("PASS", 0)) * 100
        fail_prob = float(probability_map.get("FAIL", 0)) * 100

        st.divider()
        st.subheader("Prediction Result")

        result_col, pass_col, fail_col = st.columns(3)

        with result_col:
            if prediction == "PASS":
                st.success("✅ Predicted Result: PASS")
            else:
                st.error("⚠️ Predicted Result: FAIL")

        with pass_col:
            st.metric("PASS Probability", f"{pass_prob:.2f}%")
            st.progress(int(round(pass_prob)))

        with fail_col:
            st.metric("FAIL Probability", f"{fail_prob:.2f}%")
            st.progress(int(round(fail_prob)))

        st.subheader("Student Details")
        st.write(
            f"**G1:** {g1} | **G2:** {g2} | "
            f"**Study Time:** {studytime} | "
            f"**Past Failures:** {failures} | "
            f"**Absences:** {absences}"
        )

        st.subheader("Academic Guidance")

        if fail_prob >= 60:
            st.warning(
                "Higher estimated risk of failing. "
                "Create a regular study schedule, revise weak topics, "
                "and seek help from a teacher."
            )
        elif fail_prob >= 30:
            st.info(
                "Moderate estimated risk. "
                "Practise regularly, revise previous lessons, "
                "and monitor progress."
            )
        else:
            st.success(
                "Lower estimated risk based on this model. "
                "Continue regular revision and practice."
            )

        st.caption(
            "This is a prototype. Other model inputs are taken from "
            "the first dataset record. Probabilities are estimates, "
            "not guarantees."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
