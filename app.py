import streamlit as st
import numpy as np
import pandas as pd
from joblib import load as joblib_load
from datetime import datetime

# ================== Konfigurasi Halaman ==================
st.set_page_config(
    page_title="TemanHire Hiring Decision Predictor",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ====== CSS halus untuk polish ======
st.markdown("""
<style>
/* tipis-tipis estetika */
.small-caption { font-size: 0.85rem; opacity: 0.85; }
.section-card { padding: 1rem; border: 1px solid rgba(125,125,125,0.18); border-radius: 0.75rem; }
.result-card { padding: 1.25rem; border-radius: 0.85rem; border: 1px solid rgba(125,125,125,0.18); }
.kpill { padding: .25rem .5rem; border-radius: 999px; border: 1px solid rgba(120,120,120,.25); }
.footer { opacity: .65; font-size: .85rem; text-align:center; margin-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# ================== Sidebar ==================
with st.sidebar:
    st.header("Settings")
    serious_mode = st.toggle("Serious Mode", value=True)
    show_debug = st.toggle("Tampilkan tab Debug", value=False)
    model_file = st.text_input("Nama file model", value="model.pkl", help="Samakan dengan berkas yang kamu upload")
    scaler_file = st.text_input("Nama file scaler", value="scaler.pkl", help="Samakan dengan berkas yang kamu upload")

    st.markdown("---")
    st.subheader("About")
    st.markdown(
        "Prediksi keputusan hiring berbasis AI/ML.\n\n"
        "Pastikan **urutan & nama fitur** identik dengan yang digunakan saat training."
    )

# ================== Header ==================
st.markdown(
    """
    <h1>
        <span style="color:#000000;">Hiring Prediction by Teman</span><span style="color:#0097b2;">Hire.</span>
    </h1>
    """,
    unsafe_allow_html=True
)
st.caption("Input kandidat & dapatkan prediksi keputusan hiring secara instan.")


# ================== Load Artefak ==================
@st.cache_resource
def load_joblib(path: str):
    return joblib_load(path)

load_ok = True
try:
    scaler = load_joblib(scaler_file)
    model = load_joblib(model_file)
    st.toast("Scaler & Model berhasil dimuat ✅", icon="✅")
except Exception as e:
    load_ok = False
    st.error("Gagal memuat scaler / model. Pastikan file berada di folder yang sama dengan app.py atau path sesuai.")
    with st.expander("Detail error"):
        st.code(str(e))

if not load_ok:
    st.stop()

# ================== Pemetaan Kategori ==================
edu_labels = ["Bachelor (S1)", "Bachelor (Univ Top Dunia ex:MIT,Oxford)", "Master", "PhD"]
edu_to_int = {edu_labels[0]: 1, edu_labels[1]: 2, edu_labels[2]: 3, edu_labels[3]: 4}

rs_labels = ["Agresif (HR Ganas)", "Moderat", "Pasif (Stecu)"]
rs_to_int = {rs_labels[0]: 1, rs_labels[1]: 2, rs_labels[2]: 3}

exp_labels = ["Junior", "Mid", "Senior"]

# ================== Kolom Fitur Final (HARUS sama dg training) ==================
FINAL_COLS = [
    "InterviewScore", "SkillScore", "PersonalityScore",
    "EducationLevel_2", "EducationLevel_3", "EducationLevel_4",
    "RecruitmentStrategy_2", "RecruitmentStrategy_3",
    "ExperienceLevel_Mid", "ExperienceLevel_Senior",
]

# Optional: guard ringan untuk bantu deteksi mismatch (tidak semua model punya attr ini)
if hasattr(model, "n_features_in_"):
    expected = len(FINAL_COLS)
    if model.n_features_in_ != expected:
        st.warning(
            f"Model mengharapkan {model.n_features_in_} fitur, "
            f"namun pipeline inference menyiapkan {expected}. Pastikan kolom sama persis."
        )

# ================== Tabs Utama ==================
tabs = [" Input", " Hasil"]
if show_debug:
    tabs.append("🧪 Debug")
tab_input, tab_output, *rest_tabs = st.tabs(tabs)
tab_debug = rest_tabs[0] if rest_tabs else None

# ================== Form Input ==================
with tab_input:
    st.subheader("Form Input Kandidat")
    with st.container():
        c0 = st.columns([2, 1, 1, 1])
        with c0[0]:
            candidate_name = st.text_input("Nama Kandidat", value="Agung Har!")
        with c0[1]:
            interview_score = st.number_input("InterviewScore (0-100)", min_value=0, max_value=100, value=70, step=1)
        with c0[2]:
            skill_score = st.number_input("SkillScore (0-100)", min_value=0, max_value=100, value=75, step=1)
        with c0[3]:
            personality_score = st.number_input("PersonalityScore (0-100)", min_value=0, max_value=100, value=72, step=1)

        c1 = st.columns(3)
        with c1[0]:
            education_level = st.selectbox("EducationLevel", edu_labels, index=1)
        with c1[1]:
            recruitment_strategy = st.selectbox("RecruitmentStrategy", rs_labels, index=1)
        with c1[2]:
            experience_level = st.selectbox("ExperienceLevel", exp_labels, index=0)

    submit = st.button("Prediksi", use_container_width=True)

# ================== Prediksi ==================
def build_feature_row():
    # Scaling numerik (dalam urutan yang SAMA dengan training scaler)
    X_num = np.array([[interview_score, skill_score, personality_score]], dtype=float)
    scaled = scaler.transform(X_num)
    s_interview, s_skill, s_personality = scaled[0, 0], scaled[0, 1], scaled[0, 2]

    # OHE manual sesuai skema training (basis: Edu=1, RS=1, Exp=Junior)
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

    feature_row = {
        "InterviewScore": s_interview,
        "SkillScore": s_skill,
        "PersonalityScore": s_personality,
        **ohe_cols
    }
    df_features = pd.DataFrame([feature_row], columns=FINAL_COLS)
    return df_features

def get_label_map(serious: bool):
    if serious:
        return {0: "Tidak Lolos", 1: "Lolos"}
    # fun mode
    return {
        0: "❌ Tidak Diterima Lanjut Nganggur",
        1: "✅ Diterima Menjadi Anggota DPR (Developer Product Rakyat)"
    }

if "history" not in st.session_state:
    st.session_state["history"] = []

if submit:
    try:
        df_features = build_feature_row()
        y_pred = model.predict(df_features)[0]
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = float(model.predict_proba(df_features)[0][1])

        # Simpan ke riwayat
        st.session_state["history"].append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "Nama": candidate_name,
            "Pred": int(y_pred),
            "Proba_1": y_proba,
            **df_features.iloc[0].to_dict(),
        })

        # ================== Output ==================
        with tab_output:
            label_map = get_label_map(serious_mode)
            verdict = label_map.get(int(y_pred), str(y_pred))

            st.subheader("📊 Hasil Prediksi")
            colA, colB = st.columns([1.2, 1])
            with colA:
                st.markdown(f'<div class="result-card"><h3 style="margin:0;">{candidate_name}</h3><div class="kpill" style="margin:.5rem 0;">Experience: {experience_level} • Edu: {education_level} • Strategy: {recruitment_strategy}</div><p style="font-size:1.15rem;margin:.35rem 0 0 0;"><b>Keputusan:</b> {verdict}</p></div>', unsafe_allow_html=True)
            with colB:
                if y_proba is not None:
                    st.metric("Probabilitas Lolos", f"{y_proba:.2%}")
                    st.progress(max(0.0, min(1.0, y_proba)))

            st.markdown("**Riwayat Prediksi (sesi ini)**")
            hist_df = pd.DataFrame(st.session_state["history"])
            st.dataframe(hist_df, use_container_width=True, hide_index=True)

            # Download tombol
            st.download_button(
                "⬇️ Unduh hasil terakhir (JSON)",
                data=pd.Series(st.session_state["history"][-1]).to_json(indent=2),
                file_name=f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

    except Exception as e:
        st.error("Terjadi error saat melakukan prediksi.")
        if tab_debug:
            with tab_debug:
                st.code(str(e))

# ================== Tab Debug (opsional) ==================
if tab_debug:
    st.subheader("🧪 Debug")
    st.write("Nilai mentah & hasil scaling (baris terakhir):")
    st.write("- InterviewScore / SkillScore / PersonalityScore:", interview_score, skill_score, personality_score)
    try:
        df_features_dbg = build_feature_row()
        st.dataframe(df_features_dbg, use_container_width=True)
    except Exception as e:
        st.code(str(e))

# ================== Footer ==================
st.markdown('<div class="footer">© 4Kings Prototype for demo purpose only. Validate with domain experts & HR policy.</div>', unsafe_allow_html=True)
