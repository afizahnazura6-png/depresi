import streamlit as st
import pandas as pd
import joblib
import pickle

# =========================
# Load Model
# =========================
model = joblib.load("model.joblib")
encoders = pickle.load(open("encoders.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

# =========================
# Konfigurasi Halaman
# =========================
st.set_page_config(
    page_title="Prediksi Depresi Mahasiswa",
    page_icon="🧠",
    layout="wide"
)

# =========================
# Styling
# =========================
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #2c3e50;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #7f8c8d;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<p class="title">🧠 Prediksi Depresi Mahasiswa</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Prediksi risiko depresi berdasarkan kondisi akademik dan mental</p>',
    unsafe_allow_html=True
)

st.markdown("---")

# =========================
# Mapping Kategori
# =========================
pressure_map = {
    "Sangat Rendah": 1,
    "Rendah": 2,
    "Netral": 3,
    "Tinggi": 4,
    "Sangat Tinggi": 5
}

study_map = {
    "< 2 jam": 1,
    "2 - 4 jam": 3,
    "5 - 7 jam": 6,
    "8 - 10 jam": 9,
    "> 10 jam": 12
}

satisfaction_map = {
    "Sangat Tidak Puas": 1,
    "Tidak Puas": 2,
    "Netral": 3,
    "Puas": 4,
    "Sangat Puas": 5
}

# =========================
# Input User
# =========================
col1, col2 = st.columns(2)

with col1:
    umur = st.number_input(
        "Umur",
        min_value=15,
        max_value=60,
        value=21
    )

    academic_choice = st.selectbox(
        "Tingkatan Tekanan Akademik",
        list(pressure_map.keys())
    )

    financial_choice = st.selectbox(
        "Tekanan Finansial",
        list(pressure_map.keys())
    )

    study_choice = st.selectbox(
        "Jumlah Belajar/Bekerja",
        list(study_map.keys())
    )

with col2:
    suicidal = st.selectbox(
        "Pernah memiliki suicidal thoughts?",
        ["No", "Yes"]
    )

    diet = st.selectbox(
        "Kebiasaan Pola Makan",
        ["Healthy", "Moderate", "Unhealthy"]
    )

    satisfaction_choice = st.selectbox(
        "Tingkatan Kepuasan Belajar",
        list(satisfaction_map.keys())
    )

# =========================
# Konversi ke Angka
# =========================
tekanan_akademik = pressure_map[academic_choice]
tekanan_finansial = pressure_map[financial_choice]
jam_belajar = study_map[study_choice]
kepuasan_belajar = satisfaction_map[satisfaction_choice]

# Encoding categorical
suicidal_encoded = encoders[
    "Have you ever had suicidal thoughts ?"
].transform([suicidal])[0]

diet_encoded = encoders[
    "Kebiasaan Pola Makan"
].transform([diet])[0]

# =========================
# Prediksi
# =========================
if st.button("Prediksi", use_container_width=True):

    input_data = pd.DataFrame([{
        "Have you ever had suicidal thoughts ?": suicidal_encoded,
        "Tingkatan Tekanan Akademik": tekanan_akademik,
        "Tekanan Finansial": tekanan_finansial,
        "Jumlah Belajar/Bekerja": jam_belajar,
        "Kebiasaan Pola Makan": diet_encoded,
        "Tingkatan Kepuasan Belajar": kepuasan_belajar,
        "Umur": umur
    }])

    input_data = input_data[features]

    prediction = model.predict(input_data)[0]

    st.markdown("---")

    if prediction == 1:
        st.error("⚠ Berisiko Mengalami Depresi")
        st.warning(
            "Disarankan mencari dukungan dari keluarga, teman, atau tenaga profesional."
        )
    else:
        st.success("✅ Risiko Depresi Rendah")
        st.info("Pertahankan pola hidup dan keseimbangan aktivitas Anda.")