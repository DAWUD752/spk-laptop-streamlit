# =========================================================
# IMPORT LIBRARY
# =========================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="SPK Pemilihan Laptop",
    layout="wide"
)


# =========================================================
# LOAD DATASET & SIMPAN KE SESSION
# =========================================================
@st.cache_data
def load_data():
    return pd.read_csv("dataset_laptop_50_data.xlsx")

if "data" not in st.session_state:
    st.session_state.data = load_data()


# =========================================================
# FUNGSI TOPSIS
# =========================================================
def topsis(df, weights, impacts):
    X = df.values.astype(float)

    # Normalisasi
    norm = np.sqrt((X ** 2).sum(axis=0))
    R = X / norm

    # Normalisasi berbobot
    V = R * weights

    # Solusi ideal positif & negatif
    ideal_pos = np.where(impacts == 1, V.max(axis=0), V.min(axis=0))
    ideal_neg = np.where(impacts == 1, V.min(axis=0), V.max(axis=0))

    # Jarak ke solusi ideal
    D_pos = np.sqrt(((V - ideal_pos) ** 2).sum(axis=1))
    D_neg = np.sqrt(((V - ideal_neg) ** 2).sum(axis=1))

    # Nilai preferensi
    score = D_neg / (D_pos + D_neg)

    return score


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("Kontrol Aplikasi")

top_n = st.sidebar.slider(
    "Tampilkan Top-N Laptop",
    min_value=5,
    max_value=20,
    value=10
)


# =========================================================
# JUDUL HALAMAN
# =========================================================
st.title("Sistem Pendukung Keputusan Pemilihan Laptop")
st.markdown("""
Aplikasi ini menggunakan metode **AHP** untuk menentukan bobot kriteria  
dan **TOPSIS** untuk melakukan perangkingan laptop terbaik.
""")


# =========================================================
# TAB MENU (DUA MODE)
# =========================================================
tab1, tab2 = st.tabs([
    "📊 Hasil Otomatis",
    "➕ Input Data Laptop (Interaktif)"
])


# =========================================================
# ======================= TAB 1 ===========================
# MODE HASIL OTOMATIS
# =========================================================
with tab1:

    st.subheader("Dataset Laptop (Lengkap)")
    st.dataframe(st.session_state.data)

    # -------------------------------
    # VISUALISASI SCATTER
    # -------------------------------
    st.subheader("Scatter Plot (Price vs RAM)")
    fig1, ax1 = plt.subplots()
    ax1.scatter(
        st.session_state.data["Price"],
        st.session_state.data["RAM"]
    )
    ax1.set_xlabel("Price")
    ax1.set_ylabel("RAM")
    st.pyplot(fig1)

    # -------------------------------
    # VISUALISASI HISTOGRAM
    # -------------------------------
    st.subheader("Histogram Harga Laptop")
    fig2, ax2 = plt.subplots()
    ax2.hist(
        st.session_state.data["Price"],
        bins=15
    )
    ax2.set_xlabel("Price")
    ax2.set_ylabel("Frekuensi")
    st.pyplot(fig2)

    # -------------------------------
    # VISUALISASI HEATMAP
    # -------------------------------
    st.subheader("Heatmap Korelasi Kriteria")

    criteria_cols = ["Price", "RAM", "Storage", "CPU_Speed", "Weight"]
    corr = st.session_state.data[criteria_cols].corr()

    fig3, ax3 = plt.subplots()
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax3
    )
    st.pyplot(fig3)

    # -------------------------------
    # PERHITUNGAN TOPSIS
    # -------------------------------
    st.subheader("Hasil Perangkingan TOPSIS")

    # Bobot dari AHP (contoh hasil Excel)
    weights = np.array([0.35, 0.25, 0.18, 0.12, 0.10])

    # Impact: 1 = benefit, 0 = cost
    impacts = np.array([0, 1, 1, 1, 0])

    scores = topsis(
        st.session_state.data[criteria_cols],
        weights,
        impacts
    )

    result = st.session_state.data.copy()
    result["Skor Preferensi"] = scores
    result["Ranking"] = result["Skor Preferensi"].rank(ascending=False)

    result = result.sort_values("Ranking")

    st.dataframe(result.head(top_n))

    # -------------------------------
    # BAR CHART RANKING
    # -------------------------------
    st.subheader("Grafik Top Laptop")

    fig4, ax4 = plt.subplots()
    ax4.barh(
        result.head(top_n)["Brand"],
        result.head(top_n)["Skor Preferensi"]
    )
    ax4.invert_yaxis()
    ax4.set_xlabel("Skor Preferensi")
    st.pyplot(fig4)


# =========================================================
# ======================= TAB 2 ===========================
# MODE INTERAKTIF (INPUT USER)
# =========================================================
with tab2:

    st.subheader("Tambah Data Laptop Baru")

    with st.form("form_input"):
        brand = st.text_input("Brand")
        price = st.number_input("Price", min_value=0)
        ram = st.number_input("RAM", min_value=1)
        storage = st.number_input("Storage", min_value=1)
        cpu = st.number_input("CPU Speed (GHz)", min_value=0.1)
        weight = st.number_input("Weight (Kg)", min_value=0.1)

        submit = st.form_submit_button("Tambah Data")

    if submit:
        new_data = {
            "Brand": brand,
            "Price": price,
            "RAM": ram,
            "Storage": storage,
            "CPU_Speed": cpu,
            "Weight": weight
        }

        st.session_state.data = pd.concat(
            [st.session_state.data, pd.DataFrame([new_data])],
            ignore_index=True
        )

        st.success("Data laptop berhasil ditambahkan!")
