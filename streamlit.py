import altair as alt
import pandas as pd
import streamlit as st
import numpy as np

st.set_page_config(page_title="ICU EDA Dashboard", layout="wide")
st.title("ICU EDA Dashboard")

#Load & preprocess data
@st.cache_data 
def load_data():
    df = pd.read_csv("./data/training_v2.csv")
    
    # Clean and convert BMI
    df["bmi"] = df["bmi"].replace("NA", np.nan)
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")

    prob_cols = ["apache_4a_hospital_death_prob", "apache_4a_icu_death_prob"]
    for col in prob_cols:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan

    #can add other preprocessing steps here
    
    return df

df = load_data()

# -------- SECTION 1 — Demographic Distribution --------

# -------- SECTION 2 — Numeric Predictor Distribution by Outcome --------

# Define the interested numerical predictors 
predictor_options = {
    "Age": "age",
    "Body Mass Index (BMI)": "bmi",
    "Day 1 Mean Blood Pressure (max)": "d1_mbp_max",
    "Day 1 Temperature (max)": "d1_temp_max",
    "APACHE IV Hospital Death Probability": "apache_4a_hospital_death_prob",
    "APACHE IV ICU Death Probability": "apache_4a_icu_death_prob",
    "Pre-ICU Length of Stay (days)": "pre_icu_los_days",
    "APACHE Creatinine": "creatinine_apache",
    "APACHE BUN": "bun_apache",
    "APACHE FiO₂": "fio2_apache",
    "APACHE Glucose": "glucose_apache",
    "APACHE Heart Rate": "heart_rate_apache",
}

st.header("Numeric Predictor Distribution by Outcome")

predictor_label = st.selectbox("Choose predictor", list(predictor_options.keys()))
predictor = predictor_options[predictor_label]

sub_df = df[[predictor, "hospital_death"]].dropna().copy()

outcome_map = {0: "Alive", 1: "Deceased"}
sub_df["Outcome"] = sub_df["hospital_death"].astype(int).map(outcome_map)

# Legend-based interaction: click Alive/Deceased to highlight
outcome_sel = alt.selection_multi(fields=["Outcome"], bind="legend")

chart = (
    alt.Chart(sub_df)
    .mark_bar()
    .encode(
        x=alt.X(
            f"{predictor}:Q",
            bin=alt.Bin(maxbins=30),
            title=predictor_label,              
        ),
        y=alt.Y(
            "count()",
            title="Number of ICU admissions",
        ),
        color=alt.Color(
            "Outcome:N",
            legend=alt.Legend(title="Outcome"),
            scale=alt.Scale(
                domain=["Alive", "Deceased"],
                range=["#8db2e0", "#ffc4bf"] 
            ),
        ),
        opacity=alt.condition(outcome_sel, alt.value(1.0), alt.value(0.2)),
        tooltip=[
            alt.Tooltip(f"{predictor}:Q", title=predictor_label),
            alt.Tooltip("Outcome:N", title="Outcome"),
            alt.Tooltip("count():Q", title="Number of ICU admissions"),
        ],
    )
    .add_params(outcome_sel)
)

st.altair_chart(chart, use_container_width=True)

# -------- SECTION 3 — Lab Result Change in Time --------