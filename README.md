# Prediksi Realisasi Anggaran Satker

Dashboard prediksi apakah realisasi anggaran satker akan mencapai >= 95% berdasarkan indikator kinerja pelaksanaan anggaran (IKPA).

## Teknologi
- Python 3.10
- Streamlit 1.36
- scikit-learn 1.3.2 (Random Forest Classifier)

## Cara Menjalankan
1. Clone repository ini
2. Install dependencies: `pip install -r requirements.txt`
3. Jalankan: `streamlit run app.py`

## Struktur Project
```
├── app.py                  # Aplikasi utama Streamlit
├── requirements.txt        # Daftar library
├── models/
│   └── model_realisasi_anggaran.pkl  # Model Random Forest (pickle)
├── data/
│   └── realisasi_anggaran.csv        # Dataset training
└── README.md
```

## Model
- Algoritma: Random Forest Classifier (100 trees, max_depth=10)
- Dataset: 500 satker dummy (untuk pelatihan)
- Target: Realisasi anggaran >= 95% (Ya/Tidak)

## Fitur Input
- Kementerian/Lembaga, Provinsi, Tipe Satker, Jenis Belanja
- Pagu Anggaran, Jumlah Pegawai, Jumlah SPM, Revisi DIPA
- Realisasi per Triwulan (TW-1, TW-2, TW-3)
- Deviasi RPD Halaman III DIPA, Skor IKPA

## Dibuat untuk
Pelatihan Data Analitik DJPb/Anggaran - Pusdiklat Keuangan Umum
