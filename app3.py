import streamlit as st
import pandas as pd
import numpy as np
import joblib


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="HerHealth AI",
    page_icon="🌸",
    layout="centered"
)


# =========================================================
# LOAD MODELS
# =========================================================

# Ovulation model
ovulation_model = joblib.load(
    "HerHealthAI_Ovulation_model.pkl"
)

# PCOS models
pcos_model = joblib.load(
    "pcos_model.pkl"
)

pcos_imputer = joblib.load(
    "pcos_imputer.pkl"
)

pcos_scaler = joblib.load(
    "pcos_scaler.pkl"
)


# =========================================================
# TITLE
# =========================================================

st.title("🌸 HerHealth AI")

st.write(
    "AI-Based Women's Health Assessment"
)

st.divider()


# =========================================================
# MODULE SELECTION
# =========================================================

module = st.selectbox(
    "Select Health Assessment",
    [
        "Select",
        "Menstrual / Ovulation",
        "PCOS"
    ]
)


# =========================================================
# DEFAULT
# =========================================================

if module == "Select":

    st.info(
        "Please select a health assessment."
    )


# =========================================================
# OVULATION MODULE
# =========================================================

elif module == "Menstrual / Ovulation":

    st.header(
        "🩸 Menstrual / Ovulation Assessment"
    )

    st.write(
        "Enter the following information."
    )

    # -----------------------------------------------------
    # NUMERICAL FEATURES
    # -----------------------------------------------------

    cycle_length_days = st.number_input(
        "Cycle Length (days)",
        min_value=1.0,
        max_value=100.0,
        value=28.0
    )

    prev_cycle_length = st.number_input(
        "Previous Cycle Length (days)",
        min_value=1.0,
        max_value=100.0,
        value=28.0
    )

    pain_level = st.number_input(
        "Pain Level (0-10)",
        min_value=0.0,
        max_value=10.0,
        value=3.0
    )

    mood_score = st.number_input(
        "Mood Score (0-10)",
        min_value=0.0,
        max_value=10.0,
        value=5.0
    )

    stress_score_cycle = st.number_input(
        "Stress Score (0-10)",
        min_value=0.0,
        max_value=10.0,
        value=5.0
    )

    sleep_hours_cycle = st.number_input(
        "Sleep Hours",
        min_value=0.0,
        max_value=24.0,
        value=7.0
    )

    energy_level = st.number_input(
        "Energy Level (0-10)",
        min_value=0.0,
        max_value=10.0,
        value=5.0
    )

    concentration_score = st.number_input(
        "Concentration Score (0-10)",
        min_value=0.0,
        max_value=10.0,
        value=5.0
    )

    work_hours_lost = st.number_input(
        "Work/Study Hours Lost",
        min_value=0.0,
        max_value=100.0,
        value=0.0
    )

    estrogen_pgml = st.number_input(
        "Estrogen (pg/mL)",
        min_value=0.0,
        max_value=2000.0,
        value=100.0
    )

    progesterone_ngml = st.number_input(
        "Progesterone (ng/mL)",
        min_value=0.0,
        max_value=100.0,
        value=5.0
    )

    overall_health_score = st.number_input(
        "Overall Health Score (0-10)",
        min_value=0.0,
        max_value=10.0,
        value=5.0
    )

    prepared_before_period = st.selectbox(
        "Prepared Before Period?",
        [
            "No",
            "Yes"
        ]
    )

    prepared_value = (
        1
        if prepared_before_period == "Yes"
        else 0
    )

    # -----------------------------------------------------
    # CATEGORICAL FEATURES
    # -----------------------------------------------------

    flow_level = st.selectbox(
        "Flow Level",
        [
            "Light",
            "Moderate",
            "Heavy"
        ]
    )

    pms_symptoms = st.selectbox(
        "PMS Symptoms?",
        [
            "No",
            "Yes"
        ]
    )

    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    if st.button(
        "Predict Ovulation",
        type="primary"
    ):

        ovulation_input = pd.DataFrame({

            "cycle_length_days": [
                cycle_length_days
            ],

            "prev_cycle_length": [
                prev_cycle_length
            ],

            "pain_level": [
                pain_level
            ],

            "mood_score": [
                mood_score
            ],

            "stress_score_cycle": [
                stress_score_cycle
            ],

            "sleep_hours_cycle": [
                sleep_hours_cycle
            ],

            "energy_level": [
                energy_level
            ],

            "concentration_score": [
                concentration_score
            ],

            "work_hours_lost": [
                work_hours_lost
            ],

            "estrogen_pgml": [
                estrogen_pgml
            ],

            "progesterone_ngml": [
                progesterone_ngml
            ],

            "overall_health_score": [
                overall_health_score
            ],

            "prepared_before_period": [
                prepared_value
            ],

            "flow_level": [
                flow_level
            ],

            "pms_symptoms": [
                pms_symptoms
            ]
        })

        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        prediction = ovulation_model.predict(
            ovulation_input
        )[0]

        # -------------------------------------------------
        # PROBABILITY
        # -------------------------------------------------

        probabilities = (
            ovulation_model.predict_proba(
                ovulation_input
            )[0]
        )

        classes = list(
            ovulation_model.classes_
        )

        positive_index = classes.index(
            "Positive"
        )

        positive_probability = (
            probabilities[positive_index] * 100
        )

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        st.divider()

        st.subheader(
            "Prediction Result"
        )

        st.write(
            "Prediction:",
            prediction
        )

        st.metric(
            "Ovulation Positive Probability",
            f"{positive_probability:.2f}%"
        )

        # -------------------------------------------------
        # POSITIVE
        # -------------------------------------------------

        if prediction == "Positive":

            st.success(
                "🟢 Ovulation Positive"
            )

            st.info(
                "The model identified a pattern "
                "associated with ovulation."
            )

            st.caption(
                "This is an AI-based prediction and "
                "not a confirmed medical diagnosis."
            )

            st.subheader(
                "General Guidance"
            )

            tips = [

                "Continue tracking your menstrual cycle.",

                "Record your period dates and cycle length.",

                "Maintain a balanced and nutritious diet.",

                "Stay physically active.",

                "Maintain regular sleep.",

                "Stay adequately hydrated.",

                "Continue monitoring cycle-related symptoms."
            ]

            for i, item in enumerate(
                tips,
                1
            ):

                st.write(
                    f"**{i}.** {item}"
                )

        # -------------------------------------------------
        # NEGATIVE
        # -------------------------------------------------

        else:

            st.warning(
                "🔴 Ovulation Negative"
            )

            st.info(
                "The model did not identify the pattern "
                "associated with ovulation."
            )

            st.caption(
                "A negative prediction alone does not "
                "confirm an ovulation problem."
            )

            st.subheader(
                "Recommended Actions"
            )

            precautions = [

                "Continue tracking your menstrual cycle.",

                "Record period dates and cycle length.",

                "Monitor unusual changes in your cycle.",

                "Maintain a balanced and nutritious diet.",

                "Stay physically active.",

                "Maintain adequate sleep and hydration.",

                "Manage stress through healthy activities.",

                "If you are concerned about ovulation "
                "or fertility, consult a qualified "
                "healthcare professional."
            ]

            for i, item in enumerate(
                precautions,
                1
            ):

                st.write(
                    f"**{i}.** {item}"
                )


# =========================================================
# PCOS MODULE
# =========================================================

elif module == "PCOS":

    st.header(
        "🌸 PCOS Risk Assessment"
    )

    st.write(
        "Enter the required information."
    )

    # -----------------------------------------------------
    # BASIC INFORMATION
    # -----------------------------------------------------

    age = st.number_input(
        "Age (years)",
        min_value=10.0,
        max_value=80.0,
        value=25.0
    )

    weight = st.number_input(
        "Weight (Kg)",
        min_value=20.0,
        max_value=200.0,
        value=60.0
    )

    height = st.number_input(
        "Height (Cm)",
        min_value=100.0,
        max_value=220.0,
        value=160.0
    )

    height_m = height / 100

    bmi = weight / (
        height_m ** 2
    )

    st.write(
        f"**Calculated BMI:** {bmi:.2f}"
    )

    blood_group = st.number_input(
        "Blood Group Code",
        min_value=0.0,
        max_value=20.0,
        value=11.0
    )

    pulse = st.number_input(
        "Pulse Rate (bpm)",
        min_value=40.0,
        max_value=180.0,
        value=72.0
    )

    rr = st.number_input(
        "Respiratory Rate",
        min_value=8.0,
        max_value=40.0,
        value=18.0
    )

    hb = st.number_input(
        "Hemoglobin (g/dl)",
        min_value=5.0,
        max_value=20.0,
        value=12.0
    )

    # -----------------------------------------------------
    # CYCLE
    # -----------------------------------------------------

    cycle = st.selectbox(
        "Cycle Type",
        [
            "Regular",
            "Irregular"
        ]
    )

    cycle_value = (
        2
        if cycle == "Regular"
        else 4
    )

    cycle_length = st.number_input(
        "Cycle Length (days)",
        min_value=1.0,
        max_value=60.0,
        value=28.0
    )

    marriage = st.number_input(
        "Marriage Status (years)",
        min_value=0.0,
        max_value=50.0,
        value=0.0
    )

    pregnant = st.selectbox(
        "Pregnant?",
        [
            "No",
            "Yes"
        ]
    )

    pregnant_value = (
        0
        if pregnant == "No"
        else 1
    )

    abortions = st.number_input(
        "Number of Abortions",
        min_value=0.0,
        max_value=20.0,
        value=0.0
    )

    # -----------------------------------------------------
    # HORMONES
    # -----------------------------------------------------

    beta_hcg_1 = st.number_input(
        "I beta-HCG (mIU/mL)",
        min_value=0.0,
        max_value=100000.0,
        value=0.0
    )

    beta_hcg_2 = st.number_input(
        "II beta-HCG (mIU/mL)",
        min_value=0.0,
        max_value=100000.0,
        value=0.0
    )

    fsh = st.number_input(
        "FSH (mIU/mL)",
        min_value=0.0,
        max_value=100.0,
        value=5.0
    )

    lh = st.number_input(
        "LH (mIU/mL)",
        min_value=0.0,
        max_value=100.0,
        value=5.0
    )

    fsh_lh = st.number_input(
        "FSH/LH",
        min_value=0.0,
        max_value=20.0,
        value=1.0
    )

    hip = st.number_input(
        "Hip (inch)",
        min_value=20.0,
        max_value=70.0,
        value=38.0
    )

    waist = st.number_input(
        "Waist (inch)",
        min_value=20.0,
        max_value=70.0,
        value=32.0
    )

    waist_hip = st.number_input(
        "Waist:Hip Ratio",
        min_value=0.0,
        max_value=3.0,
        value=0.84
    )

    tsh = st.number_input(
        "TSH (mIU/L)",
        min_value=0.0,
        max_value=100.0,
        value=2.0
    )

    amh = st.number_input(
        "AMH (ng/mL)",
        min_value=0.0,
        max_value=50.0,
        value=3.0
    )

    prl = st.number_input(
        "PRL (ng/mL)",
        min_value=0.0,
        max_value=200.0,
        value=15.0
    )

    vit_d3 = st.number_input(
        "Vitamin D3 (ng/mL)",
        min_value=0.0,
        max_value=200.0,
        value=30.0
    )

    prg = st.number_input(
        "PRG (ng/mL)",
        min_value=0.0,
        max_value=100.0,
        value=5.0
    )

    rbs = st.number_input(
        "RBS (mg/dl)",
        min_value=0.0,
        max_value=500.0,
        value=100.0
    )

    # -----------------------------------------------------
    # SYMPTOMS
    # -----------------------------------------------------

    weight_gain = st.selectbox(
        "Weight Gain?",
        ["No", "Yes"]
    )

    hair_growth = st.selectbox(
        "Excess Hair Growth?",
        ["No", "Yes"]
    )

    skin_darkening = st.selectbox(
        "Skin Darkening?",
        ["No", "Yes"]
    )

    hair_loss = st.selectbox(
        "Hair Loss?",
        ["No", "Yes"]
    )

    pimples = st.selectbox(
        "Pimples?",
        ["No", "Yes"]
    )

    fast_food = st.selectbox(
        "Fast Food?",
        ["No", "Yes"]
    )

    exercise = st.selectbox(
        "Regular Exercise?",
        ["No", "Yes"]
    )

    bp_sys = st.number_input(
        "Systolic BP",
        min_value=60.0,
        max_value=250.0,
        value=120.0
    )

    bp_dia = st.number_input(
        "Diastolic BP",
        min_value=30.0,
        max_value=150.0,
        value=80.0
    )

    follicle_l = st.number_input(
        "Follicle Number Left",
        min_value=0.0,
        max_value=100.0,
        value=5.0
    )

    follicle_r = st.number_input(
        "Follicle Number Right",
        min_value=0.0,
        max_value=100.0,
        value=5.0
    )

    fsize_l = st.number_input(
        "Average Follicle Size Left (mm)",
        min_value=0.0,
        max_value=50.0,
        value=10.0
    )

    fsize_r = st.number_input(
        "Average Follicle Size Right (mm)",
        min_value=0.0,
        max_value=50.0,
        value=10.0
    )

    endometrium = st.number_input(
        "Endometrium (mm)",
        min_value=0.0,
        max_value=50.0,
        value=8.0
    )

    # -----------------------------------------------------
    # PCOS PREDICTION
    # -----------------------------------------------------

    if st.button(
        "Predict PCOS Risk",
        type="primary"
    ):

        expected_features = list(
            pcos_imputer.feature_names_in_
        )

        user_data = pd.DataFrame(
            np.nan,
            index=[0],
            columns=expected_features
        )

        values = {

            " Age (yrs)": age,

            "Weight (Kg)": weight,

            "Height(Cm) ": height,

            "BMI": bmi,

            "Blood Group": blood_group,

            "Pulse rate(bpm) ": pulse,

            "RR (breaths/min)": rr,

            "Hb(g/dl)": hb,

            "Cycle(R/I)": cycle_value,

            "Cycle length(days)": cycle_length,

            "Marraige Status (Yrs)": marriage,

            "Pregnant(Y/N)": pregnant_value,

            "No. of aborptions": abortions,

            "  I   beta-HCG(mIU/mL)": beta_hcg_1,

            "II    beta-HCG(mIU/mL)": beta_hcg_2,

            "FSH(mIU/mL)": fsh,

            "LH(mIU/mL)": lh,

            "FSH/LH": fsh_lh,

            "Hip(inch)": hip,

            "Waist(inch)": waist,

            "Waist:Hip Ratio": waist_hip,

            "TSH (mIU/L)": tsh,

            "AMH(ng/mL)": amh,

            "PRL(ng/mL)": prl,

            "Vit D3 (ng/mL)": vit_d3,

            "PRG(ng/mL)": prg,

            "RBS(mg/dl)": rbs,

            "Weight gain(Y/N)":
                1 if weight_gain == "Yes" else 0,

            "hair growth(Y/N)":
                1 if hair_growth == "Yes" else 0,

            "Skin darkening (Y/N)":
                1 if skin_darkening == "Yes" else 0,

            "Hair loss(Y/N)":
                1 if hair_loss == "Yes" else 0,

            "Pimples(Y/N)":
                1 if pimples == "Yes" else 0,

            "Fast food (Y/N)":
                1 if fast_food == "Yes" else 0,

            "Reg.Exercise(Y/N)":
                1 if exercise == "Yes" else 0,

            "BP _Systolic (mmHg)": bp_sys,

            "BP _Diastolic (mmHg)": bp_dia,

            "Follicle No. (L)": follicle_l,

            "Follicle No. (R)": follicle_r,

            "Avg. F size (L) (mm)": fsize_l,

            "Avg. F size (R) (mm)": fsize_r,

            "Endometrium (mm)": endometrium
        }

        for feature, value in values.items():

            if feature in user_data.columns:

                user_data.loc[0, feature] = value

        # -------------------------------------------------
        # IMPUTATION
        # -------------------------------------------------

        user_imputed = pcos_imputer.transform(
            user_data
        )

        # -------------------------------------------------
        # SCALING
        # -------------------------------------------------

        user_scaled = pcos_scaler.transform(
            user_imputed
        )

        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        pcos_prediction = pcos_model.predict(
            user_scaled
        )[0]

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        st.divider()

        st.subheader(
            "Prediction Result"
        )

        st.write(
            "Prediction:",
            pcos_prediction
        )

        # -------------------------------------------------
        # PROBABILITY
        # -------------------------------------------------

        if hasattr(
            pcos_model,
            "predict_proba"
        ):

            pcos_probability = (
                pcos_model.predict_proba(
                    user_scaled
                )[0][1] * 100
            )

            st.metric(
                "PCOS Risk Probability",
                f"{pcos_probability:.2f}%"
            )

        # -------------------------------------------------
        # PCOS POSITIVE
        # -------------------------------------------------

        if pcos_prediction == 1:

            st.warning(
                "⚠️ PCOS Risk Detected"
            )

            st.info(
                "The model identified a pattern "
                "associated with PCOS risk."
            )

            st.caption(
                "This is an AI-based risk assessment "
                "and not a confirmed diagnosis."
            )

            st.subheader(
                "Recommended Actions"
            )

            precautions = [

                "Track your menstrual cycle regularly.",

                "Monitor irregular periods and other symptoms.",

                "Maintain a balanced and nutritious diet.",

                "Stay physically active.",

                "Maintain regular sleep habits.",

                "Manage stress through healthy activities.",

                "Consult a qualified gynecologist or "
                "healthcare professional for further evaluation."
            ]

            for i, item in enumerate(
                precautions,
                1
            ):

                st.write(
                    f"**{i}.** {item}"
                )

        # -------------------------------------------------
        # PCOS NEGATIVE
        # -------------------------------------------------

        else:

            st.success(
                "✅ PCOS Risk Not Detected"
            )

            st.info(
                "The model did not identify a high-risk "
                "PCOS pattern from the information provided."
            )

            st.subheader(
                "General Health Tips"
            )

            tips = [

                "Continue maintaining a balanced diet.",

                "Stay physically active.",

                "Maintain regular sleep habits.",

                "Track your menstrual cycle.",

                "Maintain healthy lifestyle habits.",

                "Consult a healthcare professional "
                "if concerning symptoms persist."
            ]

            for i, item in enumerate(
                tips,
                1
            ):

                st.write(
                    f"**{i}.** {item}"
                )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "HerHealth AI provides AI-based assessment support "
    "and is not a substitute for professional medical diagnosis."
)