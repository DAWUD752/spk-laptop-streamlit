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
# LOAD DATASET
# =========================================================
@st.cache_data
def load_data():
    return pd.read_csv("laptop.csv")

if "data" not in st.session_state:
    st.session_state.data = load_data()

# =========================================================
# FUNGSI AHP
# =========================================================
def ahp(pairwise_matrix):
    col_sum = pairwise_matrix.sum(axis=0)
    norm_matrix = pairwise_matrix / col_sum
    weights = norm_matrix.mean(axis=1)
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

top_n = st.sidebar.slider("Tampilkan Top-N Laptop", 5, 20, 10)

# ---------------- AHP INTERAKTIF ----------------
st.sidebar.subheader("AHP - Bobot Kriteria")

criteria = ["Price", "RAM", "Storage", "CPU_Speed", "Weight"]
n = len(criteria)

pairwise = np.ones((n, n))

for i in range(n):
    for j in range(i + 1, n):
        val = st.sidebar.slider(
            f"{criteria[i]} vs {criteria[j]}",
            1/9, 9.0, 1.0, step=0.1
        )
        pairwise[i, j] = val
        pairwise[j, i] = 1 / val

weights_ahp = ahp(pairwise)

st.sidebar.markdown("### Bobot AHP")
for c, w in zip(criteria, weights_ahp):
    st.sidebar.write(f"{c}: **{w:.3f}**")

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
# TAB MENU
# =========================================================
tab1, tab2 = st.tabs([
    "📊 Hasil Otomatis",
    "➕ Input Data Laptop"
])

# =========================================================
# TAB 1 - HASIL OTOMATIS
# =========================================================
with tab1:

    st.subheader("Dataset Laptop (Lengkap)")
    st.dataframe(st.session_state.data, use_container_width=True)

    # ---------------- SCATTER ----------------
    st.subheader("Scatter Plot (Price vs RAM)")
    fig1, ax1 = plt.subplots()
    ax1.scatter(
        st.session_state.data["Price"],
        st.session_state.data["RAM"]
    )
    ax1.set_xlabel("Price")
    ax1.set_ylabel("RAM")
    st.pyplot(fig1)

    # ---------------- HISTOGRAM ----------------
    st.subheader("Histogram Harga Laptop")
    fig2, ax2 = plt.subplots()
    ax2.hist(st.session_state.data["Price"], bins=15)
    ax2.set_xlabel("Price")
    ax2.set_ylabel("Frekuensi")
    st.pyplot(fig2)

    # ---------------- HEATMAP ----------------
    st.subheader("Heatmap Korelasi Kriteria")
    corr = st.session_state.data[criteria].corr()
    fig3, ax3 = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax3)
    st.pyplot(fig3)

    # ---------------- TOPSIS ----------------
    st.subheader("Hasil Perangkingan TOPSIS")

    impacts = np.array([0, 1, 1, 1, 0])  # cost / benefit

    scores = topsis(
        st.session_state.data[criteria],
        weights_ahp,
        impacts
    )

    result = st.session_state.data.copy()
    result["Skor Preferensi"] = scores
    result["Ranking"] = result["Skor Preferensi"].rank(ascending=False)
    result = result.sort_values("Ranking")

    st.dataframe(result.head(top_n), use_container_width=True)

    # ---------------- BAR CHART ----------------
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
# TAB 2 - INPUT DATA
# =========================================================
with tab2:

    st.subheader("Tambah Data Laptop Baru")

    with st.form("input_form"):
        brand = st.text_input("Brand")
        price = st.number_input("Price", min_value=0.0)
        ram = st.number_input("RAM (GB)", min_value=1.0)
        storage = st.number_input("Storage (GB)", min_value=1.0)
        cpu = st.number_input("CPU Speed (GHz)", min_value=0.1)
        weight = st.number_input("Weight (Kg)", min_value=0.1)
        submit = st.form_submit_button("Tambah Data")

    if submit:
        new_row = pd.DataFrame([{
            "Brand": brand,
            "Price": price,
            "RAM": ram,
            "Storage": storage,
            "CPU_Speed": cpu,
            "Weight": weight
        }])

        st.session_state.data = pd.concat(
            [st.session_state.data, new_row],
            ignore_index=True
        )

        st.success("✅ Data laptop berhasil ditambahkan!")
