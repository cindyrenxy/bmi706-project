import altair as alt
import pandas as pd
import streamlit as st
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu

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
st.header("Demographic Distribution by Ethnicity")

ethnicity_options = sorted(df["ethnicity"].dropna().unique())
selected_ethnicity = st.selectbox("Choose ethnicity", ethnicity_options)

demo_df = (
    df[df["ethnicity"] == selected_ethnicity][
        ["ethnicity", "hospital_death", "age", "gender"]
    ]
    .dropna()
    .copy()
)

outcome_map = {0: "Alive", 1: "Deceased"}
demo_df["Outcome"] = demo_df["hospital_death"].astype(int).map(outcome_map)

outcome_bar = (
    alt.Chart(demo_df)
    .mark_bar()
    .encode(
        x=alt.X("Outcome:N", title="Hospital Outcome"),
        y=alt.Y("count()", title="Number of ICU Admissions"),
        color=alt.Color(
            "Outcome:N",
            scale=alt.Scale(
            domain=["Alive", "Deceased"],
            range=["#3A5F8A", "#E76F51"],
            #range=["#4C9AFF", "#FF6F61"],  
        ),
            legend=alt.Legend(title="Outcome"),
        ),
        tooltip=[
            alt.Tooltip("Outcome:N", title="Outcome"),
            alt.Tooltip("count():Q", title="Count"),
        ],
    )
    .properties(title=f"Outcome for {selected_ethnicity}")
)

st.altair_chart(outcome_bar, use_container_width=True)

# --- BOTTOM LEFT: Age Pie with Custom Age Groups ---
demo_df["age_group"] = pd.cut(
    demo_df["age"],
    bins=[0, 20, 40, 60, 80, 200],
    labels=["0–20", "20–40", "40–60", "60–80", "80+"],
    right=False,
)

age_pie = (
    alt.Chart(demo_df)
    .transform_aggregate(count="count()", groupby=["age_group"])
    .mark_arc(innerRadius=50, outerRadius=120)
    .encode(
        theta=alt.Theta("count:Q"),
        color=alt.Color(
            "age_group:N",
            title="Age Group",
            scale=alt.Scale(
            range=[
            "#F3ECE7", 
            "#E6D3C1",  
            "#D9A679",  
            "#C97A5E",  
            "#8C3A2B", 
        ]
)
        ),
        tooltip=[
            alt.Tooltip("age_group:N", title="Age Group"),
            alt.Tooltip("count:Q", title="Count"),
        ],
    )
    .properties(title="Age Distribution")
)


gender_pie = (
    alt.Chart(demo_df)
    .transform_aggregate(count="count()", groupby=["gender"])
    .mark_arc(innerRadius=50, outerRadius=120)
    .encode(
        theta=alt.Theta("count:Q"),
        color=alt.Color(
            "gender:N",
            title="Gender",
            scale=alt.Scale(
                domain=["F", "M"],
                range=["#D9CBB6", "#7A9BBE"],  
            ),
        ),
        tooltip=[
            alt.Tooltip("gender:N", title="Gender"),
            alt.Tooltip("count:Q", title="Count"),
        ],
    )
    .properties(title="Gender Distribution")
)



col1, col2 = st.columns(2)

with col1:
    st.altair_chart(age_pie, use_container_width=True)

with col2:
    st.altair_chart(gender_pie, use_container_width=True)



# -------- SECTION 2 — Numeric Predictor Distribution by Outcome --------

# Define the interested numerical predictors 
predictor_options = {
    "Age": "age",
    "Body Mass Index (BMI)": "bmi",
    "Pre-ICU Length of Stay (days)": "pre_icu_los_days",

    "Day 1 Mean Blood Pressure (Max)": "d1_mbp_max",
    "Day 1 Temperature (Max)": "d1_temp_max",
    "Day 1 Heart Rate (Max)": "heart_rate_apache",

    "APACHE IV Hospital Death Probability": "apache_4a_hospital_death_prob",
    "APACHE IV ICU Death Probability": "apache_4a_icu_death_prob",

    "APACHE Creatinine": "creatinine_apache",
    "APACHE Blood Urea Nitrogen (BUN)": "bun_apache",
    "APACHE Glucose": "glucose_apache",
    "APACHE Fraction of Inspired Oxygen (FiO₂)": "fio2_apache",

    "Day 1 White Blood Cell Count (Max)": "d1_wbc_max",
    "Day 1 Hematocrit (Min)": "d1_hct_min",

    "Day 1 Sodium (Max)": "d1_sodium_max",
    "Day 1 Potassium (Max)": "d1_potassium_max",

    "Day 1 Platelet Count (Min)": "d1_platelets_min",

    "Day 1 Lactate (Max)": "d1_lactate_max",

    "Day 1 Creatinine (Max)": "d1_creatinine_max"
}

st.header("ICU Risk Factor Distributions by Survival Outcome")

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
lab_operations = {
    # ---- Chemistry / Metabolic Panels ----
    "Serum Albumin": "albumin_max",
    "Total Bilirubin": "bilirubin_max",
    "Blood Urea Nitrogen (BUN)": "bun_max",
    "Total Serum Calcium": "calcium_max",
    "Serum Creatinine": "creatinine_max",
    "Blood Glucose": "glucose_max",
    "Bicarbonate (HCO₃)": "hco3_max",
    
    # ---- Hematology ----
    "Hemoglobin Concentration": "hemaglobin_max",
    "Hematocrit Percentage": "hematocrit_max",
    "International Normalized Ratio (INR)": "inr_max",
    "Blood Lactate Level": "lactate_max",
    "Platelet Count": "platelets_max",
    "Serum Potassium": "potassium_max",
    "Serum Sodium": "sodium_max",
    "White Blood Cell Count (WBC)": "wbc_max",

    # ---- Respiratory / Gas Exchange ----
    "Arterial Partial Pressure of Carbon Dioxide (PaCO₂)": "arterial_pco2_max",
    "Arterial Blood pH": "arterial_ph_max",
    "Arterial Partial Pressure of Oxygen (PaO₂)": "arterial_po2_max",
    "Oxygenation Ratio (PaO₂ / FiO₂)": "pao2fio2ratio_max",

    # ---- Vital Signs ----
    "Heart Rate (Beats per Minute)": "heartrate_max",
    "Respiratory Rate (Breaths per Minute)": "resprate_max",
    "Peripheral Oxygen Saturation (SpO₂)": "spo2_max",
    "Body Temperature": "temp_max",
    "Mean Arterial Pressure (MAP)": "mbp_max",
    "Diastolic Blood Pressure": "diasbp_max",
    "Systolic Blood Pressure": "sysbp_max",
}


st.header("Change of Selected ICU Labs Operations (D1 − H1)")

# selection box for lab feature
lab_sel = st.selectbox("Choose a lab operation", list(lab_operations.keys()))
lab_key = lab_operations[lab_sel]

# extract the H1 and D1 columns for selected lab operation
h1_col = f"h1_{lab_key}"
d1_col = f"d1_{lab_key}"
lab_df = df[[h1_col, d1_col, "hospital_death", "patient_id"]].copy()
lab_df["Outcome"] = lab_df["hospital_death"].map({0: "Alive", 1: "Deceased"})
lab_df["delta"] = lab_df[d1_col] - lab_df[h1_col] # calculate delta

# statistical tests
alive = lab_df[lab_df["Outcome"]=="Alive"]["delta"].dropna()
dead  = lab_df[lab_df["Outcome"]=="Deceased"]["delta"].dropna()
t,p_welch = ttest_ind(alive, dead, equal_var=False)
u,p_mw = mannwhitneyu(alive, dead, alternative='two-sided')
st.subheader(f"Δ Distribution for **{lab_sel}**")
st.write(f"**Welch t-test** p = {p_welch:.2e}")
st.write(f"**Mann-Whitney** p = {p_mw:.2e}")

# box + swarm plot
box = (
    alt.Chart(lab_df)
    .mark_boxplot(size=50, median={'color':'white'})
    .encode(
        x=alt.X("Outcome:N", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("delta:Q", title=f"Δ {lab_sel} (D1−H1)"),
        color=alt.Color(
            "Outcome:N",
            scale=alt.Scale(domain=["Alive","Deceased"], range=["#3A5F8A", "#E76F51"])
        )
    )
)

swarm = (
    alt.Chart(lab_df)
    .mark_circle(size=40, opacity=0.55)
    .encode(
        x="Outcome:N",
        y="delta:Q",
        color=alt.Color(
            "Outcome:N",
            scale=alt.Scale(domain=["Alive","Deceased"], range=["#3A5F8A", "#E76F51"])
        ),
        tooltip=["patient_id","delta"]
    )
)

lab_plot = (box + swarm).properties(width=350, height=350)
st.altair_chart(lab_plot, use_container_width=True)
