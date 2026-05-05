import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import json
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# SELECTION CARD
# =========================
st.markdown("""
<style>
.card {
    background-color: #210F37;
    padding: 5px;
    border-radius: 10px;
    text-align: center;
    color: white;
    font-weight: bold;
    margin: 6px;
}
.card-title{
    font-size:15px;
    opacity:0.9;
}
.card-value{
    font-size:30px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("Hasil_Final.csv")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    data = joblib.load('KNN_Model.joblib')
    return data['model'], data['scaler'], data['features']

model, scaler, features_used = load_model()

# =========================
# LOAD DATA MAP
# =========================
@st.cache_data
def load_map():
    df = pd.read_csv("kabupaten_kota_cluster.csv")

    with open("Kabupaten-Kota_(Provinsi Jawa Timur).geojson") as f:
        geojson = json.load(f)

    return df, geojson

df_map, geojson = load_map()


# =========================
# SESSION STATE PAGE
# =========================
if 'page' not in st.session_state:
    st.session_state.page = "visualisasi"


# =========================
# SIDEBAR
# =========================
with st.sidebar:

    st.image(
        "logo_kominfo.png",
    )

    st.markdown("## Menu")

    if st.button("📊 Data Overview", use_container_width=True):
        st.session_state.page = "visualisasi"
        st.rerun()

    if st.button("📉 Analisis Korelasi", use_container_width=True):
        st.session_state.page = "korelasi"
        st.rerun()

    if st.button("🗺️ Peta Cluster Kemiskinan", use_container_width=True):
        st.session_state.page = "cluster"
        st.rerun()

    if st.button("🔍 Prediksi Data", use_container_width=True):
        st.session_state.page = "prediksi"
        st.rerun()

    st.markdown("---")
    st.info("💡 Gunakan halaman prediksi untuk mencoba simulasi cluster kemiskinan.")


# =========================
# HALAMAN VISUALISASI
# =========================
if st.session_state.page == "visualisasi":

    st.title("Dashboard Tingkat Kemiskinan Prov.Jawa Timur")
    st.write("Visualisasi ini menunjukkan tren data setiap varibael di kabupaten/kota jawa timur.")

## Selection Card

    st.subheader("📊 Profil Daerah")

    daerah = st.selectbox(
        "Pilih Kabupaten/Kota",
        df["Kabupaten/Kota Se Jawa Timur"].unique() 
    )

    data_daerah = df[df["Kabupaten/Kota Se Jawa Timur"] == daerah]

    gk = data_daerah["Garis Kemiskinan (Rupiah/Bulan/Kapita)"].values[0]
    p1 = data_daerah["Indeks Kedalaman Kemiskinan (P1)"].values[0]
    p2 = data_daerah["Indeks Keparahan Kemiskinan (P2)"].values[0]
    umk = data_daerah["UMK (Rupiah)"].values[0]
    tpt = data_daerah["TPT (Persen)"].values[0]
    gini = data_daerah["Gini Rasio"].values[0]
    rasio = data_daerah["Rasio UMK & GK"].values[0]
    bansos = data_daerah["Jumlah Penerima Bantuan Sosial"].values[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Garis Kemiskinan (Rupiah)</div>
            <div class="card-value">{gk:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Indeks P1</div>
            <div class="card-value">{p1}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Indeks P2</div>
            <div class="card-value">{p2}</div>
        </div>
        """, unsafe_allow_html=True)

    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">UMK (Rupiah)</div>
            <div class="card-value">{umk:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">TPT(%)</div>
            <div class="card-value">{tpt}</div>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Gini Rasio</div>
            <div class="card-value">{gini}</div>
        </div>
        """, unsafe_allow_html=True)
   
    col7, col8 = st.columns(2)

    with col7:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Rasio Kesejahteraan (UMK/GK)</div>
            <div class="card-value">{rasio:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col8:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Jumlah Bantuan Sosial Kemiskinan Ekstrim</div>
            <div class="card-value">{bansos:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
## Diagram perbandingan P1 dan P2
    st.title("Perbandingan P1 dan P2")

    fig, ax = plt.subplots(figsize=(10,5))

    ax.plot(
        df["Kabupaten/Kota Se Jawa Timur"],
        df["Indeks Kedalaman Kemiskinan (P1)"],
        marker='o',
        label="P1"
    )

    ax.plot(
        df["Kabupaten/Kota Se Jawa Timur"],
        df["Indeks Keparahan Kemiskinan (P2)"],
        marker='o',
        label="P2"
    )

    ax.set_title("Perbandingan P1 dan P2")
    ax.legend()
    ax.tick_params(axis='x', rotation=90)

    st.pyplot(fig)

### Diagram Batang P2
    # Ambil Top 5 dan Bottom 5
    top5_p1 = df.nlargest(5, "Indeks Kedalaman Kemiskinan (P1)")
    bottom5_p1 = df.nsmallest(5, "Indeks Kedalaman Kemiskinan (P1)")

    # Buat diagram batang
    fig_top = px.bar(
        top5_p1,
        y="Kabupaten/Kota Se Jawa Timur",
        x="Indeks Kedalaman Kemiskinan (P1)",
        orientation="h",
        title="Top 5 Nilai P1 Tertinggi",
        color_discrete_sequence=["green"]
    )

    fig_bottom = px.bar(
        bottom5_p1,
        y="Kabupaten/Kota Se Jawa Timur",
        x="Indeks Kedalaman Kemiskinan (P1)",
        orientation="h",
        title="Bottom 5 Nilai P1 Terendah",
        color_discrete_sequence=["red"]
    )
    
    # Tampilkan berdampingan
    col1, col2 = st.columns(2)
    with col1:
     st.plotly_chart(fig_top, use_container_width=True)
    with col2:
     st.plotly_chart(fig_bottom, use_container_width=True)

### Diagram Batang P2
    # Ambil Top 5 dan Bottom 5
    top5_p2 = df.nlargest(5, "Indeks Keparahan Kemiskinan (P2)")
    bottom5_p2 = df.nsmallest(5, "Indeks Keparahan Kemiskinan (P2)")

    # Buat diagram batang
    fig_top = px.bar(
        top5_p2,
        y="Kabupaten/Kota Se Jawa Timur",
        x="Indeks Keparahan Kemiskinan (P2)",
        orientation="h",
        title="Top 5 Nilai P2 Tertinggi",
        color_discrete_sequence=["green"]
    )

    fig_bottom = px.bar(
        bottom5_p2,
        y="Kabupaten/Kota Se Jawa Timur",
        x="Indeks Keparahan Kemiskinan (P2)",
        orientation="h",
        title="Bottom 5 Nilai P2 Terendah",
        color_discrete_sequence=["red"]
    )
    
    # Tampilkan berdampingan
    col3, col4 = st.columns(2)
    with col3:
     st.plotly_chart(fig_top, use_container_width=True)
    with col4:
     st.plotly_chart(fig_bottom, use_container_width=True)


# =========================
# HALAMAN KORELASI
# =========================
elif st.session_state.page == "korelasi":
    
    st.title("Analisis Korelasi Variabel")
    st.write("Visualisasi hubungan antara indikator kemiskinan.")

    var1 = 'Indeks Keparahan Kemiskinan (P2)'
    var2 = 'Indeks Kedalaman Kemiskinan (P1)'

    fig, ax = plt.subplots(figsize=(8,6))

    sns.regplot(
        data=df,
        x=var1,
        y=var2,
        ci=None,
        scatter_kws={'color':'blue'},
        line_kws={'color':'red','linestyle':'--'},
        ax=ax
    )

    ax.set_title("Korelasi Positif antara P1 dan P2")
    ax.set_xlabel(var1)
    ax.set_ylabel(var2)
    ax.grid(True)

    st.pyplot(fig)

    # Hitung korelasi
    correlation = df[var1].corr(df[var2])

    st.metric(
        label="Koefisien Korelasi",
        value=f"{correlation:.2f}"
    )

# =========================
# HALAMAN CLUSTER
# =========================
elif st.session_state.page == "cluster":

    st.title("Peta Cluster Kemiskinan Jawa Timur 2025")
    st.write("Visualisasi peta ini menunjukkan pengelompokan kabupaten/kota berdasarkan tingkat kemiskinan.")

    fig = px.choropleth(
        df_map,
        geojson=geojson,
        locations="Kabupaten/Kota Se Jawa Timur",
        featureidkey="properties.NAME_2",
        color="Cluster",

        color_discrete_map={
            'Daerah dengan Tingkat Kemiskinan Tinggi': "red",
            'Daerah dengan Tingkat Kemiskinan Sedang': "orange",
            'Daerah dengan Tingkat Kemiskinan Rendah': "green",
        },

        hover_name="Kabupaten/Kota Se Jawa Timur"
    )

    fig.update_geos(fitbounds="locations", visible=False)

    st.plotly_chart(fig, use_container_width=True)


# =========================
# HALAMAN PREDIKSI
# =========================
elif st.session_state.page == "prediksi":

    st.title("Prediksi Cluster Kemiskinan")
    st.write("Masukkan data indikator untuk memprediksi cluster kemiskinan.")

    # INPUT USER
    garis_kemiskinan = st.number_input("Garis Kemiskinan", value=550000.0)
    umk = st.number_input("UMK", value=2500000.0)
    tpt = st.number_input("TPT (%)", value=4.5)
    p1 = st.number_input("Indeks Kedalaman Kemiskinan (P1)", value=0.7)
    p2 = st.number_input("Indeks Keparahan Kemiskinan (P2)", value=1.2)
    gini_rasio = st.number_input("Gini Rasio", value=0.324)
    rasio_umk_gk = st.number_input("Rasio UMK / Garis Kemiskinan", value=2.0)

    st.markdown("---")

    # =========================
    # PREDIKSI
    # =========================
    if st.button("🔍 Prediksi Sekarang"):

        input_data = pd.DataFrame([[
            garis_kemiskinan,
            umk,
            tpt,
            p1,
            p2,
            gini_rasio,
            rasio_umk_gk
        ]], columns=features_used)

        # normalisasi
        input_scaled = scaler.transform(input_data)

        # prediksi
        prediction = model.predict(input_scaled)[0]

        st.success(f"Hasil Prediksi Cluster: **{prediction}**")
