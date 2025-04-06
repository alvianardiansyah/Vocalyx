import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests
import base64
import urllib.parse
import altair as alt
import plotly.express as px


# Set page configuration
st.set_page_config(
    page_title="Pencarian Lirik Lagu",
    page_icon="🎵",
    layout="wide",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .result-card {
        background-color: #b0b3b8;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .song-title {
        font-size: 24px;
        font-weight: bold;
        color: #1e3a8a;
    }
    .artist-name {
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 10px;
    }
    .match-score {
        font-size: 16px;
        color: #047857;
        font-weight: bold;
    }
    .lyrics-preview {
        font-size: 16px;
        margin-top: 10px;
        border-left: 3px solid #d1d5db;
        padding-left: 10px;
        font-style: italic;
    }
    .spotify-button {
        background-color: #38f731;
        color: white;
        padding: 12px 20px;
        border-radius: 20px;
        text-decoration: none;
        font-weight: bold;
        display: inline-block;
        margin-top: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .spotify-button:hover {
        background-color: #1ed760;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        transform: translateY(-2px);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #b0b3b8;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #011447;
        color: white;
    }
    .divider {
        border-bottom: 1px solid #e5e7eb;
        margin: 20px 0;
    }
    /* Custom styling for checkboxes */
    .stCheckbox > label > div[data-testid="stMarkdownContainer"] p {
        font-size: 16px;
        font-weight: 500;
    }
    /* Charts container */
    .chart-container {
        display: flex;
        gap: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Spotify API Credentials
# Store these securely in Streamlit secrets or environment variables in production
SPOTIFY_CLIENT_ID = "ba08ed430a7941d8abb0b5aff6589cf1"
SPOTIFY_CLIENT_SECRET = "8861888773e84646a23a9dc62940d43d"

# Function to get Spotify API token
@st.cache_data(ttl=3600)  # Cache token for 1 hour
def get_spotify_token():
    auth_url = "https://accounts.spotify.com/api/token"
    auth_header = base64.b64encode(f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}".encode()).decode()
    headers = {
        "Authorization": f"Basic {auth_header}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "client_credentials"}
    
    try:
        response = requests.post(auth_url, headers=headers, data=data)
        response.raise_for_status()
        return response.json().get("access_token")
    except requests.exceptions.RequestException as e:
        st.warning(f"Tidak dapat terhubung ke Spotify API: {str(e)}")
        return None

# Function to search for a song on Spotify
def search_spotify(track_name, artist_name, token):
    if not token:
        return None
    
    # Combine track and artist for more accurate search
    query = f"{track_name} artist:{artist_name}"
    search_url = f"https://api.spotify.com/v1/search?q={urllib.parse.quote(query)}&type=track&limit=1"
    
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.get(search_url, headers=headers)
        response.raise_for_status()
        results = response.json()
        
        tracks = results.get("tracks", {}).get("items", [])
        if tracks:
            track = tracks[0]
            return {
                "name": track["name"],
                "artist": track["artists"][0]["name"],
                "album": track["album"]["name"],
                "image_url": track["album"]["images"][1]["url"] if track["album"]["images"] else None,
                "preview_url": track["preview_url"],
                "spotify_url": track["external_urls"]["spotify"],
                "popularity": track["popularity"]
            }
        return None
    except requests.exceptions.RequestException:
        return None

# Header
st.title("🎵 Sistem Pencarian Lirik Lagu")
st.markdown("Cari lagu berdasarkan potongan lirik yang kamu ingat")

# Helper function to get color based on popularity
def get_popularity_color(popularity):
    if popularity < 30:
        return "gray"
    elif popularity < 60:
        return "orange"
    else:
        return "green"

# Function for text preprocessing
@st.cache_resource
def load_preprocessors():
    factory_stop = StopWordRemoverFactory()
    stop_remover = factory_stop.create_stop_word_remover()
    factory_stem = StemmerFactory()
    stemmer = factory_stem.create_stemmer()
    return stop_remover, stemmer

stop_remover, stemmer = load_preprocessors()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove numbers & punctuation
    text = stop_remover.remove(text)      # Remove stopwords
    text = stemmer.stem(text)             # Stem words
    return text

# Load and preprocess dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_excel('indo-song.xlsx')
        df['lirik_clean'] = df['lirik'].astype(str).apply(clean_text)
        return df
    except FileNotFoundError:
        st.error("File dataset tidak ditemukan! Pastikan file dataset berada di direktori yang sama.")
        return None

# Initialize TF-IDF Vectorizer
@st.cache_resource
def create_vectorizer(df):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['lirik_clean'])
    return vectorizer, tfidf_matrix

# Sidebar for information
with st.sidebar:
    st.image("music.jpg", width=300)  # Replace with actual music icon URL
    st.header("Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini menggunakan:
    - **TF-IDF Vectorizer** untuk mengubah teks menjadi vektor
    - **Cosine Similarity** untuk mengukur kemiripan lirik
    - **Sastrawi** untuk stemming dan menghilangkan stopwords Bahasa Indonesia
    - **Spotify API** untuk menghubungkan dengan lagu di platform Spotify
    
    Cara Penggunaan:
    1. Masukkan potongan lirik yang kamu ingat
    2. Sistem akan mencari 5 lagu yang paling mirip
    3. Hasil pencarian ditampilkan berdasarkan tingkat kemiripan
    4. Putar lagu langsung dari halaman web
    """)
    
    st.markdown("---")
    
    # Spotify API status
    spotify_token = get_spotify_token()
    if spotify_token:
        st.success("✅ Terhubung dengan Spotify API")
    else:
        st.error("❌ Tidak terhubung dengan Spotify API")
        st.info("Masukkan kredensial Spotify API yang valid untuk mengaktifkan fitur ini")
    
    st.markdown("---")
    st.markdown("Dibuat oleh Kelompok 14 STKI")

# Load data
df = load_data()

if df is not None:
    vectorizer, tfidf_matrix = create_vectorizer(df)
    
    # Search function
    def cari_lagu(query):
        query_clean = clean_text(query)
        query_vec = vectorizer.transform([query_clean])
        cosine_sim = cosine_similarity(query_vec, tfidf_matrix)[0]
        
        # Get top 5 results
        top_idx = cosine_sim.argsort()[-5:][::-1]
        top_scores = cosine_sim[top_idx]
        
        # Normalize scores to probabilities (sum to 1)
        if sum(top_scores) > 0:  # Prevent division by zero
            probabilities = top_scores / sum(top_scores)
        else:
            probabilities = top_scores
            
        hasil = df.iloc[top_idx].copy()
        hasil['similarity'] = top_scores
        hasil['probability'] = probabilities * 100  # Convert to percentage
        
        return hasil
    
    # Create tabs for different sections
    tabs = st.tabs(["🔍 Pencarian", "📊 Statistik", "❓ Bantuan"])
    
    with tabs[0]:  # Search Tab
        # Search input
        query = st.text_input("Masukkan potongan lirik:", placeholder="Contoh: aku ingin kembali ke masa lalu")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            search_button = st.button("🔍 Cari Lagu", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("🗑️ Reset", type="secondary", use_container_width=False)
        
        # Reset search results
        if clear_button:
            query = ""
            st.experimental_rerun()
        
        # Search results
        if query and search_button:
            st.markdown("---")
            hasil = cari_lagu(query)
            
            if not hasil.empty and hasil['similarity'].max() > 0:
                st.subheader(f"Hasil Pencarian untuk: '{query}'")
                
                # Sort by probability first
                hasil_sorted = hasil.sort_values(by='probability', ascending=False)
                
                # Create a dictionary to store Spotify popularity data for pie chart
                spotify_popularity_data = {}
                
                # Get Spotify data for all results
                if spotify_token:
                    for i, row in hasil_sorted.iterrows():
                        spotify_data = search_spotify(row['title'], row['artist'], spotify_token)
                        if spotify_data:
                            spotify_popularity_data[row['title']] = spotify_data['popularity']
                        else:
                            spotify_popularity_data[row['title']] = 0
                
                # Create two columns for side-by-side charts
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    # Create horizontal bar chart with better styling
                    fig, ax = plt.subplots(figsize=(8, 5))
                    
                    # Create shortened titles for better visualization
                    title_pendek = [f"{title[:25]}..." if len(title) > 25 else title for title in hasil_sorted['title']]
                    
                    # Create colorful bars with gradient
                    colors = plt.cm.YlGnBu(np.linspace(0.7, 1, len(title_pendek)))
                    bars = ax.barh(title_pendek, hasil_sorted['probability'], color=colors)
                    
                    # Adding percentage labels
                    for i, v in enumerate(hasil_sorted['probability']):
                        ax.text(v + 1, i, f"{v:.1f}%", va='center', fontweight='bold')
                    
                    # Add styling
                    ax.set_xlabel('Probabilitas Kecocokan (%)', fontsize=12, fontweight='bold')
                    ax.set_title('Tingkat Kemiripan Lagu', fontsize=16, fontweight='bold', pad=20)
                    ax.grid(axis='x', linestyle='--', alpha=0.7)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    
                    # Set background color
                    fig.patch.set_facecolor('#f5f5f5')
                    ax.set_facecolor('#f5f5f5')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with col_chart2:
                    if spotify_popularity_data:
                        # Create pie chart for popularity
                        popularity_df = pd.DataFrame({
                            'title': list(spotify_popularity_data.keys()),
                            'popularity': list(spotify_popularity_data.values())
                        })
                        
                        fig = px.pie(
                            popularity_df, 
                            values='popularity', 
                            names='title',
                            title='Popularitas Lagu di Spotify',
                            color_discrete_sequence=px.colors.sequential.Plasma
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        fig.update_layout(
                            legend_title="Judul Lagu",
                            legend=dict(orientation="h", y=-0.2),
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Tidak dapat mengambil data popularitas dari Spotify")
                
                # Show results as cards
                for i, row in hasil_sorted.iterrows():
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"""
                            <div class="result-card">
                                <div class="song-title">{row['title']}</div>
                                <div class="artist-name">oleh {row['artist']}</div>
                                <div class="match-score">Kecocokan: {row['probability']:.1f}%</div>
                                <div class="lyrics-preview">{row['lirik'][:250]}...</div>
                            """, unsafe_allow_html=True)
                            
                            # Get Spotify data if token is available
                            spotify_data = None
                            if spotify_token:
                                spotify_data = search_spotify(row['title'], row['artist'], spotify_token)
                            
                            if spotify_data:
                                # Only show audio player if preview URL exists and is not None
                                if spotify_data.get('preview_url') is not None:
                                    st.audio(spotify_data['preview_url'], format='audio/mp3')
                                else:
                                    # Instead of showing a warning, just provide Spotify link
                                    st.info("Dengarkan lagu ini langsung di Spotify")
                                    
                                # Always provide Spotify link for full song
                                st.markdown(f"""
                                <a href="{spotify_data['spotify_url']}" target="_blank" class="spotify-button">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-spotify" viewBox="0 0 16 16" style="display: inline-block; vertical-align: text-top; margin-right: 4px;">
                                        <path d="M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0zm3.669 11.538a.498.498 0 0 1-.686.165c-1.879-1.147-4.243-1.407-7.028-.77a.499.499 0 0 1-.222-.973c3.048-.696 5.662-.397 7.77.892a.5.5 0 0 1 .166.686zm.979-2.178a.624.624 0 0 1-.858.205c-2.15-1.321-5.428-1.704-7.972-.932a.625.625 0 0 1-.362-1.194c2.905-.881 6.517-.454 8.986 1.063a.624.624 0 0 1 .206.858zm.084-2.268C10.154 5.56 5.9 5.419 3.438 6.166a.748.748 0 1 1-.434-1.432c2.825-.857 7.523-.692 10.492 1.07a.747.747 0 1 1-.764 1.288z"/>
                                    </svg>
                                    Dengarkan lagu lengkap di Spotify
                                </a>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown('<span style="color: #6b7280;">Lagu tidak ditemukan di Spotify</span>', unsafe_allow_html=True)
                                
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            # Show full lyrics in expander
                            with st.expander("Lihat Lirik Lengkap"):
                                st.markdown(row['lirik'].replace('\n', '<br>'), unsafe_allow_html=True)
                                
                            # Highlight matched terms
                            if st.checkbox(f"Analisis Kecocokan untuk '{row['title']}'", key=f"checkbox_{i}"):
                                st.write("Kata kunci yang cocok:")
                                query_terms = set(clean_text(query).split())
                                song_terms = set(row['lirik_clean'].split())
                                common_terms = query_terms.intersection(song_terms)
                                
                                if common_terms:
                                    st.code(', '.join(common_terms))
                                else:
                                    st.info("Tidak ada kata yang sama persis, tetapi secara kontekstual lirik memiliki kesamaan.")
                        
                        # Show Spotify album art if available
                        with col2:
                            if spotify_token:
                                spotify_data = search_spotify(row['title'], row['artist'], spotify_token)
                                if spotify_data and spotify_data.get('image_url'):
                                    st.image(spotify_data['image_url'], width=150)
                                    
                                    # Add popularity visualization
                                    popularity = spotify_data['popularity']
                                    
                                    # Create a custom progress bar for popularity
                                    col_pop1, col_pop2 = st.columns([3, 7])
                                    with col_pop1:
                                        st.write("Popularitas:")
                                    with col_pop2:
                                        progress_color = get_popularity_color(popularity)
                                        st.progress(popularity/100, text=f"{popularity}/100")
            else:
                st.error("❌ Tidak ditemukan lagu yang cocok dengan lirik tersebut.")
                st.info("Tips: Coba gunakan kata kunci yang lebih umum atau periksa ejaan.")
    
    with tabs[1]:  # Statistics Tab
        st.header("Statistik Dataset")
        
        # Basic dataset statistics
        st.subheader("Informasi Dataset")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Jumlah Lagu", f"{len(df)}")
        with col2:
            st.metric("Jumlah Artis", f"{df['artist'].nunique()}")
        with col3:
            # Average lyrics length
            avg_length = df['lirik'].apply(lambda x: len(str(x).split())).mean()
            st.metric("Rata-rata Panjang Lirik", f"{int(avg_length)} kata")
        
        # Artist distribution
        st.subheader("Distribusi Artis Populer")
        top_artists = df['artist'].value_counts().head(10)
        
        fig = px.bar(
            x=top_artists.index, 
            y=top_artists.values,
            labels={'x': 'Artis', 'y': 'Jumlah Lagu'},
            color=top_artists.values,
            color_continuous_scale=px.colors.sequential.Viridis
        )
        fig.update_layout(title_text='Artis dengan Lagu Terbanyak')
        st.plotly_chart(fig, use_container_width=True)
        
        # Word cloud visualization section has been removed as requested
    
    with tabs[2]:  # Help Tab
        st.header("Bantuan Penggunaan")
        
        # FAQ Accordion
        with st.expander("Bagaimana cara mencari lagu?"):
            st.write("""
            1. Masukkan potongan lirik yang kamu ingat pada kolom pencarian
            2. Klik tombol "Cari Lagu"
            3. Sistem akan menampilkan 5 lagu yang memiliki kemiripan tertinggi
            4. Untuk mendengarkan lagu, gunakan pemutar audio yang tersedia langsung di halaman
            """)
        
        with st.expander("Bagaimana algoritma pencarian bekerja?"):
            st.write("""
            Aplikasi ini menggunakan kombinasi teknologi berikut:
            
            1. **Preprocessing**: Teks lirik dibersihkan menggunakan Sastrawi untuk menghilangkan stopword dan melakukan stemming
            2. **TF-IDF**: Setiap lirik diubah menjadi vektor dengan teknik Term Frequency-Inverse Document Frequency
            3. **Cosine Similarity**: Kemiripan antara query dengan lirik dihitung menggunakan metode ini
            4. **Normalisasi**: Skor kemiripan dinormalisasi menjadi persentase untuk memudahkan pemahaman
            """)
        
        with st.expander("Mengapa beberapa lagu tidak ditemukan di Spotify?"):
            st.write("""
            Beberapa kemungkinan penyebabnya:
            
            1. Judul atau artis dalam database tidak sama persis dengan yang ada di Spotify
            2. Lagu tersebut tidak tersedia di platform Spotify
            3. Terjadi kendala koneksi dengan API Spotify
            
            Solusi:
            - Coba cari secara manual di aplikasi Spotify dengan judul dan artis yang tertera
            """)
        
        with st.expander("Tips untuk hasil pencarian yang lebih baik"):
            st.write("""
            - Gunakan frasa lirik yang unik atau khas
            - Periksa ejaan kata-kata dalam lirik yang dimasukkan
            - Hindari penggunaan kata-kata umum seperti "aku", "kamu", dll.
            - Masukkan setidaknya 3-5 kata untuk hasil yang lebih spesifik
            - Jika tidak menemukan hasil yang diinginkan, coba variasi dari lirik yang kamu ingat
            """)

else:
    st.warning("Aplikasi tidak dapat dimulai karena dataset tidak ditemukan.")
    
    # For demo purposes if the dataset is not available
    st.info("""
    Demo mode: Untuk mencoba aplikasi ini, Anda memerlukan file CSV dengan kolom:
    - judul: Judul lagu
    - penyanyi: Nama penyanyi/band
    - lirik: Lirik lagu lengkap
    
    Silakan siapkan file 'lirik_lagu.csv' dan jalankan aplikasi kembali.
    """)
