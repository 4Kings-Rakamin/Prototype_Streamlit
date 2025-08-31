import streamlit as st
import numpy as np
import pandas as pd
from joblib import load as joblib_load

# ----------------- Konfigurasi halaman -----------------
st.set_page_config(page_title="4Kings FinPro 3", layout="centered")
st.title("AI Based Hiring Decision Predictor by 4Kings")
st.caption("Input → Scaling → OHE → Prediksi menggunakan model8.pkl (RandomForestClassifier, hubungi agung untuk tes 40 model lainnya) ## Disclaimer Ini prototype bukan hasil final (Ver : 0.1.5/ N2.0.2, P2.3.2, SC)")

# ----------------- Load scaler & model -----------------
@st.cache_resource
def load_joblib(path: str):
    return joblib_load(path)

try:
    scaler = load_joblib("scaler.pkl")
    model = load_joblib("model.pkl")
    st.success("Scaler & Model berhasil dimuat ✅")
except Exception as e:
    st.error("Gagal memuat scaler/model. Pastikan file ada di folder yang sama dengan app.py")
    st.code(str(e))
    st.stop()

# ----------------- Pemetaan kategori -----------------
edu_labels = ["Bachelor (S1)", "Bachelor (Univ Top Dunia ex:MIT,Oxford)", "Master", "PhD"]
edu_to_int = {edu_labels[0]: 1, edu_labels[1]: 2, edu_labels[2]: 3, edu_labels[3]: 4}

rs_labels = ["Agresif (HR Ganas)", "Moderat", "Pasif (Stecu)"]
rs_to_int = {rs_labels[0]: 1, rs_labels[1]: 2, rs_labels[2]: 3}

exp_labels = ["Junior", "Mid", "Senior"]

# ----------------- Form input -----------------
with st.form("form_input"):
    st.subheader("Form Input Kandidat")

    candidate_name = st.text_input("Nama Kandidat", value="Agung Har!")

    c1, c2, c3 = st.columns(3)
    with c1:
        interview_score = st.number_input("InterviewScore", min_value=0, max_value=100, value=70, step=1)
    with c2:
        skill_score = st.number_input("SkillScore", min_value=0, max_value=100, value=75, step=1)
    with c3:
        personality_score = st.number_input("PersonalityScore", min_value=0, max_value=100, value=72, step=1)

    c4, c5, c6 = st.columns(3)
    with c4:
        education_level = st.selectbox("EducationLevel", edu_labels, index=1)
    with c5:
        recruitment_strategy = st.selectbox("RecruitmentStrategy", rs_labels, index=1)
    with c6:
        experience_level = st.selectbox("ExperienceLevel", exp_labels, index=0)

    submitted = st.form_submit_button("🔮 Prediksi")

# ----------------- Transformasi + Prediksi -----------------
if submitted:
    # Scaling
    X_num = np.array([[interview_score, skill_score, personality_score]], dtype=float)
    scaled = scaler.transform(X_num)
    s_interview, s_skill, s_personality = scaled[0, 0], scaled[0, 1], scaled[0, 2]

    # OHE manual
    ohe_cols = {
        "EducationLevel_2": 0,
        "EducationLevel_3": 0,
        "EducationLevel_4": 0,
        "RecruitmentStrategy_2": 0,
        "RecruitmentStrategy_3": 0,
        "ExperienceLevel_Mid": 0,
        "ExperienceLevel_Senior": 0,
    }

    edu_code = edu_to_int[education_level]
    if edu_code >= 2:
        ohe_cols[f"EducationLevel_{edu_code}"] = 1

    rs_code = rs_to_int[recruitment_strategy]
    if rs_code >= 2:
        ohe_cols[f"RecruitmentStrategy_{rs_code}"] = 1

    if experience_level == "Mid":
        ohe_cols["ExperienceLevel_Mid"] = 1
    elif experience_level == "Senior":
        ohe_cols["ExperienceLevel_Senior"] = 1

    # Final feature row
    feature_row = {
        "InterviewScore": s_interview,
        "SkillScore": s_skill,
        "PersonalityScore": s_personality,
        **ohe_cols
    }

    final_cols = [
        "InterviewScore", "SkillScore", "PersonalityScore",
        "EducationLevel_2", "EducationLevel_3", "EducationLevel_4",
        "RecruitmentStrategy_2", "RecruitmentStrategy_3",
        "ExperienceLevel_Mid", "ExperienceLevel_Senior",
    ]
    df_features = pd.DataFrame([feature_row], columns=final_cols)

    # ----------------- Prediksi -----------------
    y_pred = model.predict(df_features)[0]
    y_proba = model.predict_proba(df_features)[0][1] if hasattr(model, "predict_proba") else None

    # Tampilkan hasil
    label_map = {0: "❌ Tidak Diterima Lanjut Nganggur", 1: "✅ Diterima Menjadi Anggota DPR (Developer Product Rakyat)"}
    st.subheader("📊 Hasil Prediksi")
    st.write(f"**{candidate_name}: {label_map.get(int(y_pred), y_pred)}**")


    if y_proba is not None:
        st.write(f"**Probabilitas Diterima:** {y_proba:.2%}")
        st.progress(float(y_proba))

    with st.expander("Detail fitur (Racikan 4kings YTTA)"):
        st.dataframe(df_features, use_container_width=True)
