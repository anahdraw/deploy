import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =============================================
# KONFIGURASI HALAMAN
# =============================================
st.set_page_config(
    page_title="Prediksi Realisasi Anggaran - DJPb",
    page_icon="📊",
    layout="wide",
)

# =============================================
# LOAD MODEL
# =============================================
@st.cache_resource
def load_model():
    with open("models/model_realisasi_anggaran.pkl", "rb") as f:
        artifacts = pickle.load(f)
    return artifacts

artifacts = load_model()
model = artifacts["model"]

# Daftar pilihan dari encoder
KEMENTERIAN = artifacts["kementerian_classes"]
PROVINSI = artifacts["provinsi_classes"]
TIPE_SATKER = artifacts["tipe_classes"]
JENIS_BELANJA = artifacts["jenis_classes"]

# =============================================
# HEADER
# =============================================
st.title("📊 Prediksi Realisasi Anggaran Satker")
st.markdown(
    "Aplikasi ini memprediksi apakah **realisasi anggaran satker akan mencapai ≥ 95%** "
    "berdasarkan indikator kinerja pelaksanaan anggaran (IKPA)."
)
st.markdown("---")

# =============================================
# SIDEBAR - INPUT MANUAL
# =============================================
st.sidebar.header("📝 Input Data Satker")
st.sidebar.markdown("Masukkan data satker yang ingin diprediksi:")

# Kategorikal
nama_kementerian = st.sidebar.selectbox("Kementerian/Lembaga", KEMENTERIAN)
provinsi = st.sidebar.selectbox("Provinsi", PROVINSI)
tipe_satker = st.sidebar.selectbox("Tipe Satker", TIPE_SATKER)
jenis_belanja = st.sidebar.selectbox("Jenis Belanja Utama", JENIS_BELANJA)

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Data Anggaran")

# Numerik
pagu_miliar = st.sidebar.number_input(
    "Pagu Anggaran (Miliar Rp)", min_value=0.1, max_value=1000.0, value=50.0, step=5.0
)
jumlah_pegawai = st.sidebar.number_input(
    "Jumlah Pegawai", min_value=1, max_value=1000, value=100, step=10
)
jumlah_spm = st.sidebar.number_input(
    "Jumlah SPM Terbit", min_value=1, max_value=500, value=50, step=5
)
revisi_dipa = st.sidebar.number_input(
    "Jumlah Revisi DIPA", min_value=0, max_value=10, value=1, step=1
)

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Realisasi Per Triwulan (%)")

realisasi_tw1 = st.sidebar.slider("Realisasi TW-1 (%)", 0.0, 40.0, 15.0, 0.5)
realisasi_tw2 = st.sidebar.slider("Realisasi TW-2 (%)", 0.0, 70.0, 40.0, 0.5)
realisasi_tw3 = st.sidebar.slider("Realisasi TW-3 (%)", 0.0, 95.0, 70.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Indikator Kinerja")

deviasi_rpd = st.sidebar.slider(
    "Deviasi RPD Hal. III DIPA (%)", 0.0, 50.0, 10.0, 0.5
)
skor_ikpa = st.sidebar.slider("Skor IKPA", 50.0, 100.0, 85.0, 0.5)

# =============================================
# ENCODE INPUT
# =============================================
le_kementerian = artifacts["le_kementerian"]
le_provinsi = artifacts["le_provinsi"]
le_tipe = artifacts["le_tipe"]
le_jenis = artifacts["le_jenis"]

input_data = pd.DataFrame([{
    "pagu_miliar": pagu_miliar,
    "jumlah_pegawai": jumlah_pegawai,
    "jumlah_spm": jumlah_spm,
    "revisi_dipa": revisi_dipa,
    "realisasi_tw1_persen": realisasi_tw1,
    "realisasi_tw2_persen": realisasi_tw2,
    "realisasi_tw3_persen": realisasi_tw3,
    "deviasi_rpd_persen": deviasi_rpd,
    "skor_ikpa": skor_ikpa,
    "kementerian_encoded": le_kementerian.transform([nama_kementerian])[0],
    "provinsi_encoded": le_provinsi.transform([provinsi])[0],
    "tipe_encoded": le_tipe.transform([tipe_satker])[0],
    "jenis_encoded": le_jenis.transform([jenis_belanja])[0],
}])

# =============================================
# MAIN CONTENT
# =============================================
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("📋 Ringkasan Input")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Pagu Anggaran", f"Rp {pagu_miliar:.1f} M")
    col_b.metric("Jumlah Pegawai", f"{jumlah_pegawai}")
    col_c.metric("Skor IKPA", f"{skor_ikpa:.1f}")

    col_d, col_e, col_f = st.columns(3)
    col_d.metric("Realisasi TW-1", f"{realisasi_tw1:.1f}%")
    col_e.metric("Realisasi TW-2", f"{realisasi_tw2:.1f}%")
    col_f.metric("Realisasi TW-3", f"{realisasi_tw3:.1f}%")

    st.markdown("")
    info_data = {
        "Parameter": ["Kementerian", "Provinsi", "Tipe Satker", "Jenis Belanja",
                       "Jumlah SPM", "Revisi DIPA", "Deviasi RPD"],
        "Nilai": [nama_kementerian, provinsi, tipe_satker, jenis_belanja,
                  jumlah_spm, revisi_dipa, f"{deviasi_rpd:.1f}%"],
    }
    st.dataframe(pd.DataFrame(info_data), use_container_width=True, hide_index=True)

with col_right:
    st.subheader("🔮 Hasil Prediksi")

    # Prediksi
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    prob_tercapai = probability[1] * 100
    prob_tidak = probability[0] * 100

    if prediction == 1:
        st.success(f"### ✅ Realisasi Diprediksi TERCAPAI ≥ 95%")
    else:
        st.error(f"### ❌ Realisasi Diprediksi TIDAK Tercapai")

    st.markdown("")

    col_p1, col_p2 = st.columns(2)
    col_p1.metric("Probabilitas Tercapai", f"{prob_tercapai:.1f}%")
    col_p2.metric("Probabilitas Tidak", f"{prob_tidak:.1f}%")

    # Progress bar visual
    st.markdown("**Confidence Level:**")
    st.progress(prob_tercapai / 100)

    # Rekomendasi
    st.markdown("---")
    st.subheader("💡 Rekomendasi")

    rekomendasi = []
    if realisasi_tw3 < 70:
        rekomendasi.append("⚠️ Realisasi TW-3 masih di bawah 70%. Percepat penyerapan anggaran di Q4.")
    if deviasi_rpd > 15:
        rekomendasi.append("⚠️ Deviasi RPD tinggi. Perbaiki perencanaan penarikan dana (Hal. III DIPA).")
    if skor_ikpa < 80:
        rekomendasi.append("⚠️ Skor IKPA rendah. Perhatikan indikator-indikator IKPA yang masih kurang.")
    if revisi_dipa > 3:
        rekomendasi.append("⚠️ Terlalu banyak revisi DIPA. Tingkatkan kualitas perencanaan awal.")
    if jumlah_spm < 20:
        rekomendasi.append("⚠️ Jumlah SPM sangat rendah. Pastikan proses pencairan berjalan lancar.")

    if not rekomendasi:
        st.info("✅ Semua indikator dalam kondisi baik. Pertahankan kinerja!")
    else:
        for r in rekomendasi:
            st.warning(r)

# =============================================
# SECTION: FEATURE IMPORTANCE
# =============================================
st.markdown("---")
st.subheader("📊 Faktor-Faktor yang Mempengaruhi Prediksi")

feature_names_display = {
    "pagu_miliar": "Pagu Anggaran",
    "jumlah_pegawai": "Jumlah Pegawai",
    "jumlah_spm": "Jumlah SPM",
    "revisi_dipa": "Revisi DIPA",
    "realisasi_tw1_persen": "Realisasi TW-1",
    "realisasi_tw2_persen": "Realisasi TW-2",
    "realisasi_tw3_persen": "Realisasi TW-3",
    "deviasi_rpd_persen": "Deviasi RPD",
    "skor_ikpa": "Skor IKPA",
    "kementerian_encoded": "Kementerian",
    "provinsi_encoded": "Provinsi",
    "tipe_encoded": "Tipe Satker",
    "jenis_encoded": "Jenis Belanja",
}

importances = pd.DataFrame({
    "Faktor": [feature_names_display.get(f, f) for f in artifacts["feature_cols"]],
    "Importance": model.feature_importances_,
}).sort_values("Importance", ascending=True)

st.bar_chart(importances.set_index("Faktor"))

# =============================================
# FOOTER
# =============================================
st.markdown("---")
st.caption(
    "📌 **Disclaimer**: Aplikasi ini menggunakan dummy data untuk keperluan pelatihan. "
    "Model: Random Forest Classifier | Dataset: 500 satker | "
    "Dibuat untuk Pelatihan Data Analitik DJPb/Anggaran."
)
