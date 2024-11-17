import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import numpy as np

# Setel konfigurasi halaman
st.set_page_config(page_title="Analisis Hero MLBB", layout="wide")

# Inisialisasi session state
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'processed_data' not in st.session_state:
    st.session_state['processed_data'] = None
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "Unggah Dataset"

# Fungsi menu utama menggunakan kamus halaman
def main():
    st.title("Analisis Hero Mobile Legends Bang Bang")

    pages = {
        "Unggah Dataset": upload_dataset,
        "Preprocessing Data": preprocess_data,
        "Analisis Data": analysis_data,
        "Visualisasi Data": visualization_data
    }
    
    # Sidebar dengan tombol menggantikan selectbox
    st.sidebar.title("Navigasi")
    for page in pages.keys():
        if st.sidebar.button(page):
            st.session_state['current_page'] = page
    
    # Eksekusi halaman yang dipilih
    pages[st.session_state['current_page']]()

# Fungsi Unggah Dataset
def upload_dataset():
    st.header("Unggah Dataset")
    
    # Opsi untuk mengunggah file
    uploaded_file = st.file_uploader("Pilih file CSV", type="csv")
    if uploaded_file is not None:
        st.session_state['data'] = pd.read_csv(uploaded_file)
        st.success("Data berhasil diunggah!")
    
    if st.session_state['data'] is not None:
        st.write("Pratinjau data:")
        st.write(st.session_state['data'].head())

# Fungsi Preprocessing Data
def preprocess_data():
    st.header("Preprocessing Data")
    
    if st.session_state['data'] is None:
        st.warning("Harap unggah dataset terlebih dahulu!")
        return
    
    # Tampilkan data asli
    st.subheader("Data Asli")
    st.dataframe(st.session_state['data'])
    
    # Tampilkan informasi dasar
    st.subheader("Informasi Dasar")
    basic_info = st.session_state['data'].describe()
    st.dataframe(basic_info)
    
    # Tampilkan informasi nilai yang hilang
    st.subheader("Pemeriksaan Nilai Hilang")
    
    # Buat dua kolom
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Nilai Null:")
        null_values = st.session_state['data'].isnull().sum().reset_index()
        null_values.columns = ['Kolom', 'Jumlah Null']
        st.dataframe(null_values)
    
    with col2:
        st.write("Nilai Nol:")
        zero_values = (st.session_state['data'] == 0).sum().reset_index()
        zero_values.columns = ['Kolom', 'Jumlah Nol']
        st.dataframe(zero_values)
    
    # Buat data yang diproses
    processed_data = st.session_state['data'].copy()
    
    # Tangani nilai yang hilang
    processed_data = processed_data.fillna(0)
    
    # Normalisasi kolom numerik
    numerical_cols = ['defense_overall', 'offense_overall', 'skill_effect_overall', 
                     'difficulty_overall', 'win_rate', 'pick_rate', 'ban_rate']
    
    scaler = MinMaxScaler()
    processed_data[numerical_cols] = scaler.fit_transform(processed_data[numerical_cols])
    
    st.session_state['processed_data'] = processed_data
    
    # Tampilkan data yang diproses
    st.subheader("Data yang Diproses")
    st.dataframe(processed_data)
    
    st.success("Preprocessing data selesai!")

# Fungsi lainnya tetap sama, seperti analysis_data(), hero_statistics(), 
# filter_role(), team_performance(), hero_recommendations(), match_prediction(), 
# visualization_data()

def analysis_data():
    st.header("Analisis Data")
    
    if st.session_state['processed_data'] is None:
        st.warning("Harap lakukan preprocessing data terlebih dahulu!")
        return
    
    analysis_option = st.selectbox(
        "Pilih Jenis Analisis",
        ["Statistik Hero", "Filter Berdasarkan Role", "Analisis Performa Tim", "Rekomendasi Hero", "Prediksi Hasil Pertandingan"]
    )
    
    if analysis_option == "Statistik Hero":
        hero_statistics()
    elif analysis_option == "Filter Berdasarkan Role":
        filter_role()
    elif analysis_option == "Analisis Performa Tim":
        team_performance()
    elif analysis_option == "Rekomendasi Hero":
        hero_recommendations()
    elif analysis_option == "Prediksi Hasil Pertandingan":
        match_prediction()

def hero_statistics():
    st.subheader("Statistik Hero")
    data = st.session_state['processed_data']
    
    # Tampilkan statistik hero
    stats = data[['hero_name', 'win_rate', 'pick_rate', 'ban_rate']].sort_values('win_rate', ascending=False)
    st.write(stats)

def filter_role():
    st.subheader("Filter Berdasarkan Role")
    data = st.session_state['processed_data']
    
    # Pilih role
    selected_role = st.selectbox("Pilih Role", data['role'].unique())
    
    # Filter data
    filtered_data = data[data['role'] == selected_role]
    st.write(filtered_data[['hero_name', 'win_rate', 'pick_rate', 'ban_rate']])

def team_performance():
    st.subheader("Analisis Performa Tim")
    data = st.session_state['processed_data']
    
    # Buat analisis komposisi tim
    roles = data['role'].value_counts()
    fig = px.pie(values=roles.values, names=roles.index, title="Distribusi Role")
    st.plotly_chart(fig)

def hero_recommendations():
    st.subheader("Rekomendasi Hero")
    data = st.session_state['processed_data']
    
    # Dapatkan hero teratas berdasarkan win rate
    top_heroes = data.nlargest(5, 'win_rate')[['hero_name', 'role', 'win_rate']]
    st.write("5 Hero Teratas Berdasarkan Win Rate:")
    st.write(top_heroes)

def match_prediction():
    st.subheader("Prediksi Hasil Pertandingan")
    data = st.session_state['processed_data']
    
    # Prediksi sederhana berdasarkan rata-rata win rate tim
    st.write("Pilih 5 hero untuk setiap tim:")
    
    # Pilihan Tim 1
    team1_heroes = st.multiselect("Hero Tim 1", data['hero_name'].tolist(), max_selections=5)
    
    # Pilihan Tim 2
    team2_heroes = st.multiselect("Hero Tim 2", data['hero_name'].tolist(), max_selections=5)
    
    if len(team1_heroes) == 5 and len(team2_heroes) == 5:
        team1_wr = data[data['hero_name'].isin(team1_heroes)]['win_rate'].mean()
        team2_wr = data[data['hero_name'].isin(team2_heroes)]['win_rate'].mean()
        
        st.write(f"Probabilitas kemenangan Tim 1: {team1_wr:.2f}%")
        st.write(f"Probabilitas kemenangan Tim 2: {team2_wr:.2f}%")

def visualization_data():
    st.header("Visualisasi Data")
    
    if st.session_state['processed_data'] is None:
        st.warning("Harap lakukan preprocessing data terlebih dahulu!")
        return
    
    data = st.session_state['processed_data']
    
    # Atribut Hero - Grafik Garis
    st.subheader("Atribut Hero")
    selected_hero = st.selectbox("Pilih Hero", data['hero_name'].tolist())
    
    hero_data = data[data['hero_name'] == selected_hero]
    
    # Buat grafik garis untuk atribut hero
    attributes = ['defense_overall', 'offense_overall', 'skill_effect_overall', 'difficulty_overall']
    
    fig_line = go.Figure()
    
    fig_line.add_trace(go.Scatter(
        x=attributes,
        y=hero_data[attributes].values.flatten(),
        mode='lines+markers',
        name=selected_hero
    ))
    
    fig_line.update_layout(
        title=f"Atribut Hero untuk {selected_hero}",
        xaxis_title="Atribut",
        yaxis_title="Nilai",
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig_line)
    
    # Tambahkan Analisis Klaster K-means
    st.subheader("Analisis Klaster Hero")
    
    # Pilih fitur untuk klastering
    features = ['defense_overall', 'offense_overall', 'skill_effect_overall', 'difficulty_overall']
    X = data[features].values
    
    # Lakukan klastering k-means
    n_clusters = st.slider("Pilih jumlah klaster", min_value=2, max_value=4, value=3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(X)
    
    # Visualisasikan klaster menggunakan PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Buat DataFrame untuk pemetaan
    plot_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Cluster': data['Cluster'],
        'Hero': data['hero_name'],
        'Role': data['role']
    })
    
    # Buat scatter plot
    fig_cluster = px.scatter(
        plot_df,
        x='PC1',
        y='PC2',
        color='Cluster',
        hover_data=['Hero', 'Role'],
        title='Klaster Hero Berdasarkan Atribut'
    )
    
    fig_cluster.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig_cluster)
    
    # Tampilkan analisis klaster sebagai tabel
    st.subheader("Karakteristik Klaster")
    for cluster in range(n_clusters):
        cluster_heroes = data[data['Cluster'] == cluster]
        st.write(f"Klaster {cluster}:")
        
        # Tampilkan karakteristik klaster sebagai tabel
        st.write("Karakteristik rata-rata:")
        st.dataframe(cluster_heroes[features].mean().to_frame().T)  # Tampilkan karakteristik rata-rata
        
        st.write("Hero dalam klaster ini:")
        st.dataframe(cluster_heroes[['hero_name', 'role']])  # Tampilkan hero dalam klaster ini sebagai tabel
        
        st.write("---")


if __name__ == "__main__":
    main()