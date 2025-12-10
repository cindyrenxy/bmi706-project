# Interactive ICU EDA Dashboard

## 📊 Overview
This project presents an **interactive exploratory data analysis dashboard** built using the **WiDS Datathon 2020 ICU dataset**. Our goal is to visualize early clinical signals associated with **in-hospital mortality** and provide interpretable insights to support critical-care research and decision-making.

---

## 🗂 Dataset
The dataset originates from MIT’s **GOSSIS (Global Open Source Severity of Illness Score)** initiative.

**Key characteristics:**
- **130,000+ ICU admissions** from **200+ hospitals worldwide**
- **185 clinical features**, including:
  - Demographics  
  - Comorbidities  
  - Vitals and lab measurements  
  - APACHE/GCS severity scores  
  - First-hour (H1) and first-day (D1) summaries  
- **Target variable:** `hospital_death`  
  - `1` → patient died during hospitalization  
  - `0` → patient survived  

This dataset captures a diverse, real-world ICU population suitable for exploratory analysis and clinical insight generation.

---

## 🖥 Dashboard Features

### 1. Demographic Exploration
- Visualize mortality outcomes by **ethnicity, gender, and age groups**
- Interactive filtering to compare survival patterns across subgroups

### 2. Risk-Associated Variable Distributions
- Compare key numerical predictors (e.g., **BMI, APACHE scores, vitals**) across outcomes
- Histograms and density plots with interactive legend controls

### 3. Early Clinical Change (H1 → D1)
- Analyze temporal changes in lab measurements (**Δ = Day1 − Hour1**)
- **Box + swarm plots** to contrast survivors vs. non-survivors
- Statistical testing to evaluate difference significance:
  - **Welch’s t-test** for approximate normality
  - **Mann-Whitney U test** for non-parametric robustness

---

## 👥 Target Audience
Designed for **clinicians, ICU researchers, and data scientists**, this dashboard supports:

- Identifying potential **risk factors for mortality**
- Exploring trends in early ICU physiology
- Translating raw clinical data into **interpretable visual insights**

Our goal is to bridge **data-driven findings** with **practical clinical decision support** using clear and interactive visualizations.

---

