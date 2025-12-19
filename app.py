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
# LOAD DATASET (FIXED CSV FORMAT)
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv("laptop.csv", sep=";")

    numeric_cols = ["Price", "RAM", "Storage", "CPU_Speed", "Weight"]
    df[numeric_cols] = df[numeric_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    df = df.dropna()
    return df


if "data" not in st.session_state:
    st.session_state.data = load_data()


# =========================================================
# FUNGSI AHP (INTERAKTIF)
# =========================================================
def ahp_weights(pairwise_matrix):
    eigvals, eigvecs = np.linalg.eig(pairwise_matrix)
    max_index = np.argmax(eigvals.real)
    weights = eigvecs[:, max_index].real
    weights = weights / weights.sum()
    return weights


# =========================================================
# FUNGSI TOPSIS
# =========================================================
def topsis(df, weights, impacts):
    X = df.values.astype(float)

    norm = np.sqrt((X ** 2).sum(axis=0))
    R = X / norm
    V = R * weights

    ideal_pos = np.where(impacts == 1, V.max(axis=0), V.min(axis=0))
    ideal_neg = np.where(impacts == 1, V.min(axis=0), V.max(axis=0))

    D_pos = np.sqrt(((V - ideal_pos) ** 2).sum(axis=1))
    D_neg = np.sqrt(((V - ideal_neg) ** 2).sum(axis=1))

    score = D_neg / (D_pos + D_neg)
    return score


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("Kontrol Aplikasi")

top_n = st.sidebar.slider(
    "Tampilkan Top-N Laptop",
    5, 20, 10
)

st.sidebar.subheader("AHP - Bobot Kriteria")

criteria = ["Price", "RAM", "Storage", "CPU_Speed", "Weight"]
n = len(criteria)

pairwise = np.ones((n, n))

for i in range(n):
    for j in range(i + 1, n):
        val = st.sidebar.slider(
            f"{criteria[i]} vs {criteria[j]}",
            1/9, 9.0, 1.0
        )
        pairwise[i, j] = val
        pairwise[j, i] = 1 / val

weights = ahp_weights(pairwise)


# =========================================================
# JUDUL
# =========================================================
st.title("Sistem Pendukung Keputusan Pemilihan Laptop")

st.markdown("""
Metode yang digunakan:
- **AHP** → Penentuan bobot kriteria (interaktif)
- **TOPSIS** → Perangkingan laptop terbaik
""")


# =========================================================
# TAB
# =========================================================
tab1, tab2 = st.tabs(["📊 Hasil Otomatis", "➕ Input Data Laptop"])


# =========================================================
# TAB 1
# =========================================================
with tab1:
    st.subheader("Dataset Laptop (Lengkap)")
    st.dataframe(st.session_state.data)

    st.subheader("Scatter Plot (Price vs RAM)")
    fig1, ax1 = plt.subplots()
    ax1.scatter(
        st.session_state.data["Price"],
        st.session_state.data["RAM"]
    )
    ax1.set_xlabel("Price")
    ax1.set_ylabel("RAM")
    st.pyplot(fig1)

    st.subheader("Heatmap Korelasi")
    corr = st.session_state.data[criteria].corr()
    fig2, ax2 = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

    impacts = np.array([0, 1, 1, 1, 0])
    scores = topsis(
        st.session_state.data[criteria],
        weights,
        impacts
    )

    result = st.session_state.data.copy()
    result["Skor Preferensi"] = scores
    result["Ranking"] = result["Skor Preferensi"].rank(
        ascending=False
    )

    result = result.sort_values("Ranking")

    st.subheader("Hasil Ranking TOP Laptop")
    st.dataframe(result.head(top_n))


# =========================================================
# TAB 2
# =========================================================
with tab2:
    st.subheader("Tambah Data Laptop Baru")

    with st.form("form_input"):
        brand = st.text_input("Brand")
        price = st.number_input("Price", min_value=0.0)
        ram = st.number_input("RAM", min_value=1)
        storage = st.number_input("Storage", min_value=1)
        cpu = st.number_input("CPU Speed (GHz)", min_value=0.1)
        weight = st.number_input("Weight (Kg)", min_value=0.1)
        submit = st.form_submit_button("Tambah")

    if submit:
        new_data = pd.DataFrame([{
            "Brand": brand,
            "Price": price,
            "RAM": ram,
            "Storage": storage,
            "CPU_Speed": cpu,
            "Weight": weight
        }])

        st.session_state.data = pd.concat(
            [st.session_state.data, new_data],
            ignore_index=True
        )

        st.success("Data berhasil ditambahkan!")
