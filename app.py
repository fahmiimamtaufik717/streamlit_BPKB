"""
Aplikasi Dashboard Forecasting Pengeluaran BPKB — Polres Sukabumi
Model: ARIMA(0,1,1) + drift, dengan transformasi Log
Dibangun dengan Streamlit
"""

import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA
from openpyxl import Workbook

# ──────────────────────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Forecasting BPKB — Polres Sukabumi",
    page_icon="📊",
    layout="wide",
)

ORDER = (0, 1, 1)     # ARIMA(p,d,q) hasil pemilihan model terbaik (Bab IV)
TREND = "t"           # komponen drift
NAMA_BULAN = ["Januari","Februari","Maret","April","Mei","Juni",
              "Juli","Agustus","September","Oktober","November","Desember"]

# ──────────────────────────────────────────────────────────────
# DATA HISTORIS BAWAAN (Januari 2021 – Desember 2025)
# Bisa ditambah / diganti melalui fitur upload di sidebar
# ──────────────────────────────────────────────────────────────
DEFAULT_PERIODE = pd.date_range("2021-01-01", periods=60, freq="MS")
DEFAULT_NILAI = [
    2244,2445,2547,2702,2551,3064,2395,2954,3504,3125,3274,3827,
    2560,2270,2681,2649,2277,2349,2735,3948,3053,2712,3038,3753,
    2490,2813,3019,2595,3997,3073,3743,4577,3069,2900,3371,3729,
    3253,2623,3081,3550,3664,3606,3806,4010,3381,4954,4291,3653,
    2389,3130,4414,6819,6450,7369,4491,4606,4785,3588,3336,4035,
]


def get_default_df():
    return pd.DataFrame(
        {"Tanggal": DEFAULT_PERIODE, "Jumlah_Pengeluaran_BPKB": DEFAULT_NILAI}
    )


# ──────────────────────────────────────────────────────────────
# STATE AWAL
# ──────────────────────────────────────────────────────────────
if "df_data" not in st.session_state:
    st.session_state.df_data = get_default_df()
if "hasil_forecast" not in st.session_state:
    st.session_state.hasil_forecast = None


# ──────────────────────────────────────────────────────────────
# FUNGSI INTI: FIT MODEL & FORECAST
# ──────────────────────────────────────────────────────────────
def jalankan_forecast(df, n_bulan=12):
    """Fit ARIMA(0,1,1)+drift pada data (skala Log) lalu forecast n_bulan ke depan.
    Forecast & CI dijamin >= 0 karena ditransformasi balik dengan eksponensial."""
    df = df.sort_values("Tanggal").reset_index(drop=True)
    ts = pd.Series(
        df["Jumlah_Pengeluaran_BPKB"].values,
        index=pd.DatetimeIndex(df["Tanggal"], freq="MS"),
    )

    y_log = np.log(ts)
    model = ARIMA(y_log, order=ORDER, trend=TREND).fit()

    fc_obj = model.get_forecast(steps=n_bulan)
    fc_log = fc_obj.predicted_mean
    ci_log = fc_obj.conf_int(alpha=0.05)

    forecast = np.exp(fc_log)
    ci = np.exp(ci_log).clip(lower=0)  # jaring pengaman tambahan, seharusnya otomatis >= 0

    tanggal_forecast = pd.date_range(
        ts.index[-1] + pd.DateOffset(months=1), periods=n_bulan, freq="MS"
    )

    df_fc = pd.DataFrame({
        "Tanggal": tanggal_forecast,
        "Bulan": [f"{NAMA_BULAN[d.month-1]} {d.year}" for d in tanggal_forecast],
        "Forecast": forecast.values.round(0).astype(int),
        "CI_Bawah": ci.iloc[:, 0].values.round(0).astype(int),
        "CI_Atas": ci.iloc[:, 1].values.round(0).astype(int),
    })

    return ts, model, df_fc


# ──────────────────────────────────────────────────────────────
# EXPORT KE EXCEL
# ──────────────────────────────────────────────────────────────
def buat_excel(ts, df_fc):
    wb = Workbook()
    ws1 = wb.active
    ws1.title = "Data Historis"
    ws1.append(["Tanggal", "Jumlah Pengeluaran BPKB"])
    for tgl, val in ts.items():
        ws1.append([tgl.strftime("%Y-%m-%d"), int(val)])

    ws2 = wb.create_sheet("Hasil Forecast")
    ws2.append(["Bulan", "Forecast", "CI Bawah (95%)", "CI Atas (95%)"])
    for _, row in df_fc.iterrows():
        ws2.append([row["Bulan"], int(row["Forecast"]), int(row["CI_Bawah"]), int(row["CI_Atas"])])

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf


# ──────────────────────────────────────────────────────────────
# EXPORT KE PDF (grafik + tabel ringkas)
# ──────────────────────────────────────────────────────────────
def buat_pdf(ts, df_fc):
    from matplotlib.backends.backend_pdf import PdfPages

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(ts.index, ts.values, color="#2C3E50", marker="o", markersize=3,
                linewidth=1.8, label="Data Aktual")
        ax.plot(df_fc["Tanggal"], df_fc["Forecast"], color="#E74C3C", marker="s",
                markersize=5, linewidth=2, linestyle="--", label="Forecast")
        ax.fill_between(df_fc["Tanggal"], df_fc["CI_Bawah"], df_fc["CI_Atas"],
                         alpha=0.15, color="#E74C3C", label="Interval Kepercayaan 95%")
        ax.set_title("Forecasting Pengeluaran BPKB — Polres Sukabumi", fontweight="bold")
        ax.set_ylabel("Jumlah Pengeluaran BPKB")
        ax.legend()
        ax.grid(alpha=0.25, linestyle="--")
        fig.autofmt_xdate()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(11, 6))
        ax2.axis("off")
        tabel_data = df_fc[["Bulan", "Forecast", "CI_Bawah", "CI_Atas"]].values
        tabel = ax2.table(
            cellText=tabel_data,
            colLabels=["Bulan", "Forecast", "CI Bawah (95%)", "CI Atas (95%)"],
            loc="center", cellLoc="center",
        )
        tabel.auto_set_font_size(False)
        tabel.set_fontsize(9)
        tabel.scale(1, 1.6)
        ax2.set_title(f"Tabel Hasil Forecast — dibuat {datetime.now():%d %B %Y}", fontweight="bold", pad=20)
        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)

    buf.seek(0)
    return buf


# ──────────────────────────────────────────────────────────────
# SIDEBAR — INPUT DATA
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pengaturan Data")
    st.caption("Model: ARIMA(0,1,1) + drift, transformasi Log")

    st.subheader("1️⃣ Input Data Pengeluaran BPKB Terbaru")
    uploaded = st.file_uploader(
        "Unggah file CSV (kolom: Tanggal, Jumlah_Pengeluaran_BPKB)", type=["csv"]
    )
    if uploaded is not None:
        try:
            df_baru = pd.read_csv(uploaded)
            if not {"Tanggal", "Jumlah_Pengeluaran_BPKB"}.issubset(df_baru.columns):
                st.error("Kolom CSV harus bernama: Tanggal, Jumlah_Pengeluaran_BPKB")
            else:
                df_baru["Tanggal"] = pd.to_datetime(df_baru["Tanggal"], errors="coerce")
                n_invalid = df_baru["Tanggal"].isna().sum()
                df_baru = df_baru.dropna(subset=["Tanggal"])

                if df_baru.empty:
                    st.error(
                        "Kolom 'Tanggal' tidak dikenali sebagai tanggal. "
                        "Gunakan format YYYY-MM-DD, contoh: 2026-01-01."
                    )
                else:
                    df_baru["Jumlah_Pengeluaran_BPKB"] = pd.to_numeric(
                        df_baru["Jumlah_Pengeluaran_BPKB"], errors="coerce"
                    )
                    df_baru = df_baru.dropna(subset=["Jumlah_Pengeluaran_BPKB"])

                    st.session_state.df_data = (
                        pd.concat([st.session_state.df_data, df_baru])
                        .drop_duplicates(subset="Tanggal", keep="last")
                        .sort_values("Tanggal")
                        .reset_index(drop=True)
                    )
                    st.success(f"✅ {len(df_baru)} baris data berhasil ditambahkan/diperbarui.")
                    if n_invalid > 0:
                        st.warning(
                            f"⚠️ {n_invalid} baris dilewati karena kolom 'Tanggal' tidak valid "
                            "(gunakan format YYYY-MM-DD)."
                        )
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")

    st.divider()
    st.subheader("➕ Tambah 1 Data Bulanan Manual")
    with st.form("form_manual"):
        tgl_baru = st.date_input("Bulan (pilih tanggal 1)", value=None)
        nilai_baru = st.number_input("Jumlah Pengeluaran BPKB", min_value=0, step=1)
        submit_manual = st.form_submit_button("Tambahkan")
        if submit_manual and tgl_baru is not None:
            tgl_awal_bulan = pd.Timestamp(tgl_baru).replace(day=1)
            df_new_row = pd.DataFrame({"Tanggal": [tgl_awal_bulan], "Jumlah_Pengeluaran_BPKB": [nilai_baru]})
            st.session_state.df_data = (
                pd.concat([st.session_state.df_data, df_new_row])
                .drop_duplicates(subset="Tanggal", keep="last")
                .sort_values("Tanggal")
                .reset_index(drop=True)
            )
            st.success(f"✅ Data {tgl_awal_bulan:%B %Y} ditambahkan.")

    st.divider()
    n_bulan_forecast = st.slider("Jumlah bulan forecast ke depan", 1, 24, 12)

    if st.button("🔄 Reset ke Data Bawaan"):
        st.session_state.df_data = get_default_df()
        st.session_state.hasil_forecast = None
        st.rerun()


# ──────────────────────────────────────────────────────────────
# HALAMAN UTAMA
# ──────────────────────────────────────────────────────────────
st.title("📊 Dashboard Forecasting Pengeluaran BPKB")
st.markdown("**Polres Sukabumi** — Model ARIMA(0,1,1) dengan komponen drift, transformasi Log")

tab1, tab2, tab3 = st.tabs(["📈 Forecasting", "📋 Data Historis", "ℹ️ Tentang Model"])

with tab1:
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        proses = st.button("▶️ Jalankan Forecasting Otomatis", type="primary", use_container_width=True)

    if proses:
        if len(st.session_state.df_data) < 12:
            st.warning("Data historis minimal 12 bulan diperlukan untuk menjalankan model.")
        else:
            with st.spinner("Model sedang memproses data dan menghitung forecast..."):
                ts, model, df_fc = jalankan_forecast(st.session_state.df_data, n_bulan_forecast)
                st.session_state.hasil_forecast = (ts, df_fc)

    if st.session_state.hasil_forecast is not None:
        ts, df_fc = st.session_state.hasil_forecast

        st.subheader("Visualisasi Hasil Prediksi")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(ts.index, ts.values, color="#2C3E50", marker="o", markersize=3.5,
                linewidth=1.8, label="Data Aktual")
        ax.plot(df_fc["Tanggal"], df_fc["Forecast"], color="#E74C3C", marker="s",
                markersize=6, linewidth=2.2, linestyle="--", label="Forecast")
        ax.fill_between(df_fc["Tanggal"], df_fc["CI_Bawah"], df_fc["CI_Atas"],
                         alpha=0.15, color="#E74C3C", label="Interval Kepercayaan 95%")
        ax.axvline(ts.index[-1], color="gray", linestyle=":", alpha=0.7)
        ax.set_ylabel("Jumlah Pengeluaran BPKB")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.legend(loc="upper left")
        ax.grid(alpha=0.25, linestyle="--")
        fig.autofmt_xdate()
        st.pyplot(fig)

        st.subheader("Tabel Ringkasan Forecast")
        df_tampil = df_fc[["Bulan", "Forecast", "CI_Bawah", "CI_Atas"]].rename(
            columns={"CI_Bawah": "CI Bawah (95%)", "CI_Atas": "CI Atas (95%)"}
        )
        st.dataframe(df_tampil, use_container_width=True, hide_index=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Proyeksi", f"{df_fc['Forecast'].sum():,}")
        c2.metric("Rata-rata / Bulan", f"{df_fc['Forecast'].mean():,.0f}")
        c3.metric("CI Bawah Minimum", f"{df_fc['CI_Bawah'].min():,}",
                   help="Selalu ≥ 0 karena transformasi Log")

        st.subheader("Unduh Laporan Hasil")
        cdl1, cdl2 = st.columns(2)
        with cdl1:
            excel_buf = buat_excel(ts, df_fc)
            st.download_button(
                "⬇️ Unduh Excel (.xlsx)", data=excel_buf,
                file_name=f"Forecast_BPKB_{datetime.now():%Y%m%d}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with cdl2:
            pdf_buf = buat_pdf(ts, df_fc)
            st.download_button(
                "⬇️ Unduh PDF", data=pdf_buf,
                file_name=f"Forecast_BPKB_{datetime.now():%Y%m%d}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
    else:
        st.info("👈 Klik **Jalankan Forecasting Otomatis** untuk menghasilkan prediksi.")

with tab2:
    st.subheader("Data Historis Pengeluaran BPKB")
    st.dataframe(
        st.session_state.df_data.assign(
            Tanggal=st.session_state.df_data["Tanggal"].dt.strftime("%B %Y")
        ),
        use_container_width=True, hide_index=True,
    )
    st.caption(f"Total {len(st.session_state.df_data)} observasi bulanan.")

with tab3:
    st.subheader("Tentang Model")
    st.markdown("""
    **Model:** ARIMA(0,1,1) dengan komponen drift (trend), di-*fit* pada data yang telah
    ditransformasi Logaritma Natural.

    **Alasan pemilihan model** *(ringkasan dari Bab IV — Hasil dan Pembahasan)*:
    - Dipilih berdasarkan nilai BIC terendah dan akurasi data testing (MAPE) terbaik
      dibandingkan kandidat model lain, termasuk ARIMA(2,1,0).
    - Transformasi Log memastikan hasil forecast dan interval kepercayaan **selalu bernilai
      positif** (tidak mungkin negatif).
    - Komponen drift ditambahkan agar forecast jangka panjang **mengikuti tren kenaikan**
      historis, bukan flat/datar.
    - Residual model telah lolos uji Ljung-Box (white noise) dan uji normalitas
      (Jarque-Bera & Shapiro-Wilk).

    **Keterbatasan yang perlu diperhatikan:**
    - Pada evaluasi data testing, MAPE model berada pada kategori **Cukup** (20–50%,
      mengikuti Khairunnisa, Haryadi, & Audyna, 2022).
    - Sebagai pembanding, model Naive (nilai bulan sebelumnya) menunjukkan akurasi yang
      kompetitif pada data historis, sehingga hasil forecast sebaiknya digunakan sebagai
      salah satu pertimbangan perencanaan, bukan satu-satunya acuan mutlak.
    """)

st.divider()
st.caption("Aplikasi ini dibangun sebagai bagian dari tahap Deployment penelitian skripsi — "
           "Forecasting Pengeluaran BPKB menggunakan Metode ARIMA (CRISP-DM).")
