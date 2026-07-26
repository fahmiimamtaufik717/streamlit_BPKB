# Dashboard Forecasting Pengeluaran BPKB — Polres Sukabumi

Aplikasi web berbasis **Streamlit** untuk memprediksi jumlah pengeluaran Buku Pemilik Kendaraan Bermotor (BPKB) di Polres Sukabumi menggunakan model **ARIMA(0,1,1) dengan drift** (transformasi Log).

## Fitur
- Input data bulanan terbaru (upload CSV atau input manual)
- Forecasting otomatis dengan interval kepercayaan 95%
- Visualisasi grafik tren aktual vs prediksi
- Unduh hasil dalam format Excel (.xlsx) dan PDF

## Menjalankan secara lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud
1. Push repository ini ke GitHub (public atau private).
2. Buka [share.streamlit.io](https://share.streamlit.io), login dengan akun GitHub.
3. Klik **New app**, pilih repo ini, branch `main`, dan file utama `app.py`.
4. Klik **Deploy**.

## Struktur Data CSV (untuk upload)
File CSV harus memiliki kolom:
| Tanggal    | Jumlah_Pengeluaran_BPKB |
|------------|--------------------------|
| 2026-01-01 | 3500                    |

## Model
ARIMA(0,1,1) dengan komponen drift, dipilih berdasarkan nilai BIC terendah dan akurasi data testing (MAPE) pada penelitian skripsi terkait.
