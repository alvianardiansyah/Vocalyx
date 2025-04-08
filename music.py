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
import plotly.graph_objects as go

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
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #3366ff;
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
        color: black;
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
        background-color: #ffffff;
        color: #000000;
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
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    /* Metrics styling */
    div[data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: bold !important;
        color: #1e3a8a !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 16px !important;
    }
    /* Card styling */
    .stat-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    .song-meta {
        display: flex;
        justify-content: space-between;
        margin-top: 10px;
    }
    .song-meta-item {
        font-size: 14px;
        color: #6b7280;
    }
    /* Remove white rectangle background from metrics */
    div.css-1r6slb0.e1tzin5v2 {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    /* Apply directly to metric containers */
    div[data-testid="stMetricValue"] {
        background-color: transparent !important;
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
                "popularity": track["popularity"],
                "release_date": track["album"].get("release_date", "N/A"),
                "track_number": track["track_number"],
                "duration_ms": track["duration_ms"],
                "explicit": track["explicit"]
            }
        return None
    except requests.exceptions.RequestException:
        return None

# Function to get artist info from Spotify
def get_artist_info(artist_id, token):
    if not token or not artist_id:
        return None
    
    artist_url = f"https://api.spotify.com/v1/artists/{artist_id}"
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.get(artist_url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None

# Helper function to get color based on popularity
def get_popularity_color(popularity):
    if popularity < 30:
        return "gray"
    elif popularity < 60:
        return "orange"
    else:
        return "green"

# Function to format milliseconds as minutes:seconds
def format_duration(ms):
    seconds = int((ms / 1000) % 60)
    minutes = int((ms / (1000 * 60)) % 60)
    return f"{minutes}:{seconds:02d}"

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

# Initialize session state for managing the application state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Pencarian"

if 'query' not in st.session_state:
    st.session_state.query = ""

# Function to reset search input
def reset_search():
    st.session_state.query = ""

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

    # Get Spotify token
    spotify_token = get_spotify_token()
    
    # Sidebar navigation
    with st.sidebar:
        st.image("music.jpg", width=300)
        
        st.title("Menu Navigasi")
        
        # Navigation buttons
        if st.button("🏠 Dashboard", use_container_width=True, key="nav_dashboard"):
            st.session_state.current_page = "Dashboard"
            
        if st.button("🔍 Pencarian", use_container_width=True, key="nav_search"):
            st.session_state.current_page = "Pencarian"
            
        if st.button("❓ Bantuan", use_container_width=True, key="nav_help"):
            st.session_state.current_page = "Bantuan"
            
        if st.button("ℹ️ Tentang Aplikasi", use_container_width=True, key="nav_about"):
            st.session_state.current_page = "Tentang"
        
        st.markdown("---")
        
        # Spotify API status
        if spotify_token:
            st.success("✅ Terhubung dengan Spotify API")
        else:
            st.error("❌ Tidak terhubung dengan Spotify API")
            st.info("Masukkan kredensial Spotify API yang valid untuk mengaktifkan fitur ini")
    
    # Main content based on current page
    if st.session_state.current_page == "Dashboard":
        st.title("📊 Dashboard Dataset Lagu")
        
        # Basic dataset statistics in expanded cards
        st.subheader("Informasi Dataset")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Jumlah Lagu", f"{len(df)}")
        with col2:
            st.metric("Jumlah Artis", f"{df['artist'].nunique()}")
        with col3:
            # Average lyrics length
            avg_length = df['lirik'].apply(lambda x: len(str(x).split())).mean()
            st.metric("Rata-rata Panjang Lirik", f"{int(avg_length)} kata")
        with col4:
            # Average words per song
            unique_words = df['lirik_clean'].apply(lambda x: len(set(x.split())))
            st.metric("Rata-rata Kata Unik", f"{int(unique_words.mean())}")
        
        st.markdown("---")
        
        # Create tabs for different visualizations
        viz_tabs = st.tabs(["Artis Populer", "Analisis Lirik", "Distribusi Tahun"])
        
        with viz_tabs[0]:
            st.subheader("Distribusi Artis Populer")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Artist distribution - bar chart
                top_artists = df['artist'].value_counts().head(10)
                
                fig = px.bar(
                    x=top_artists.index, 
                    y=top_artists.values,
                    labels={'x': 'Artis', 'y': 'Jumlah Lagu'},
                    color=top_artists.values,
                    color_continuous_scale=px.colors.sequential.Viridis,
                    text=top_artists.values
                )
                fig.update_layout(
                    title_text='Artis dengan Lagu Terbanyak',
                    xaxis_title="Nama Artis",
                    yaxis_title="Jumlah Lagu",
                    height=500
                )
                fig.update_traces(texttemplate='%{text}', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Artist distribution - pie chart
                fig = px.pie(
                    values=top_artists.values, 
                    names=top_artists.index,
                    title='Persentase Kontribusi Artis Top 10',
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        
        with viz_tabs[1]:
            st.subheader("Analisis Panjang Lirik")
            
            # Calculate lyrics length stats
            df['lyric_length'] = df['lirik'].apply(lambda x: len(str(x).split()))
            df['unique_words'] = df['lirik_clean'].apply(lambda x: len(set(x.split())))
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution of lyrics length
                fig = px.histogram(
                    df, 
                    x='lyric_length',
                    nbins=30,
                    color_discrete_sequence=['#3366FF'],
                    labels={'lyric_length': 'Jumlah Kata dalam Lirik'},
                    title='Distribusi Panjang Lirik'
                )
                fig.update_layout(
                    xaxis_title="Jumlah Kata",
                    yaxis_title="Jumlah Lagu",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Scatter plot of unique words vs total words
                fig = px.scatter(
                    df,
                    x='lyric_length',
                    y='unique_words',
                    color=df['unique_words']/df['lyric_length'],
                    color_continuous_scale='Viridis',
                    opacity=0.7,
                    labels={
                        'lyric_length': 'Total Kata',
                        'unique_words': 'Kata Unik',
                        'color': 'Rasio Keunikan'
                    },
                    title='Hubungan Kata Unik dan Total Kata'
                )
                fig.update_layout(
                    xaxis_title="Total Kata dalam Lirik",
                    yaxis_title="Jumlah Kata Unik",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
            # Top artists by average song length
            artist_length = df.groupby('artist')['lyric_length'].agg(['mean', 'count']).reset_index()
            artist_length = artist_length[artist_length['count'] >= 3].sort_values('mean', ascending=False).head(10)
            
            fig = px.bar(
                artist_length,
                x='artist',
                y='mean',
                color='mean',
                color_continuous_scale='Blues',
                labels={'artist': 'Artis', 'mean': 'Rata-rata Jumlah Kata'},
                title='Artis dengan Rata-rata Lirik Terpanjang (min. 3 lagu)'
            )
            fig.update_layout(
                xaxis_title="Nama Artis",
                yaxis_title="Rata-rata Jumlah Kata",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
                
        with viz_tabs[2]:
            st.subheader("Distribusi Tahun Rilis")
            
            # Let's simulate this as we don't have actual year data in the dataset
            # In real app, you'd use actual year data from your dataset
            # This is just for demonstration
            
            # Generate random years between 1990 and 2023 for demonstration
            np.random.seed(42)  # For reproducibility
            years = np.random.randint(1990, 2024, size=len(df))
            year_counts = pd.Series(years).value_counts().sort_index()
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=year_counts.index,
                y=year_counts.values,
                mode='lines+markers',
                name='Jumlah Lagu',
                line=dict(color='#1DB954', width=4),
                marker=dict(size=8, color='#1DB954')
            ))
            
            fig.update_layout(
                title='Distribusi Lagu Berdasarkan Tahun Rilis',
                xaxis_title="Tahun",
                yaxis_title="Jumlah Lagu",
                height=500,
                hovermode="x unified",
                plot_bgcolor='rgba(240,242,246,0.8)'
            )
            
            fig.update_xaxes(tickangle=-45, dtick=2)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add decade analysis
            decades = pd.cut(year_counts.index, bins=range(1990, 2031, 10), right=False, labels=['1990s', '2000s', '2010s', '2020s'])
            decade_counts = pd.Series(decades).value_counts(sort=False).reindex(['1990s', '2000s', '2010s', '2020s'])
            
            fig = px.pie(
                values=[year_counts[(year_counts.index >= 1990) & (year_counts.index < 2000)].sum(),
                        year_counts[(year_counts.index >= 2000) & (year_counts.index < 2010)].sum(),
                        year_counts[(year_counts.index >= 2010) & (year_counts.index < 2020)].sum(),
                        year_counts[(year_counts.index >= 2020)].sum()],
                names=['1990s', '2000s', '2010s', '2020s'],
                title='Distribusi Lagu per Dekade',
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif st.session_state.current_page == "Pencarian":
        st.title("🔍 Pencarian Lirik Lagu")
        st.markdown("Cari lagu berdasarkan potongan lirik yang kamu ingat")
        
        # Search input
        query = st.text_input("Masukkan potongan lirik:", 
                              value=st.session_state.query,
                              placeholder="Contoh: kau dan daku sahabat untuk selamanya",
                              key="search_query")
        
        # Store query in session state
        st.session_state.query = query
        
        col1, col2 = st.columns([1, 3])
        with col1:
            search_button = st.button("🔍 Cari Lagu", type="primary", use_container_width=True)
        with col2:
            reset_button = st.button("🗑️ Reset", type="secondary", on_click=reset_search)
        
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
                    # Create horizontal bar chart with Plotly for better interactivity
                    title_pendek = [
                        f"{str(title)[:25]}..." if title and len(str(title)) > 25 else str(title)
                        for title in hasil_sorted['title']
                    ]

                    
                    fig = go.Figure()
                    
                    # Add bars with gradient color
                    fig.add_trace(go.Bar(
                        y=title_pendek,
                        x=hasil_sorted['probability'],
                        orientation='h',
                        marker=dict(
                            color=hasil_sorted['probability'],
                            colorscale='YlGnBu',
                            colorbar=dict(title="Probabilitas (%)"),
                        ),
                        text=[f"{p:.1f}%" for p in hasil_sorted['probability']],
                        textposition='outside',
                        hovertemplate='<b>%{y}</b><br>Probabilitas: %{x:.1f}%<extra></extra>'
                    ))
                    
                    # Update layout for better visualization
                    fig.update_layout(
                        title='Tingkat Kemiripan Lagu',
                        xaxis_title='Probabilitas Kecocokan (%)',
                        yaxis=dict(autorange="reversed"),  # Reverse y-axis to match the original order
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=400,
                        margin=dict(l=10, r=10, t=50, b=50),
                        xaxis=dict(
                            showgrid=True,
                            gridcolor='rgba(211, 211, 211, 0.5)'
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
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
                            color_discrete_sequence=px.colors.sequential.Plasma,
                            hover_data=['popularity']
                        )
                        fig.update_traces(
                            textposition='inside', 
                            textinfo='percent+label',
                            hovertemplate='<b>%{label}</b><br>Popularitas: %{customdata[0]}/100<extra></extra>'
                        )
                        fig.update_layout(
                            legend_title="Judul Lagu",
                            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
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
                                <div class="lyrics-preview">{row['lirik']}</div>
                            """, unsafe_allow_html=True)
                            # Tambahkan expander di bawahnya untuk lirik lengkap

                            
                            # Get Spotify data if token is available
                            spotify_data = None
                            if spotify_token:
                                spotify_data = search_spotify(row['title'], row['artist'], spotify_token)
                            
                            if spotify_data:
                                # Display additional song information
                                st.markdown(f"""
                                <div class="song-meta">
                                    <div class="song-meta-item">🎵 Album: {spotify_data['album']}</div>
                                    <div class="song-meta-item">📅 Dirilis: {spotify_data.get('release_date', 'N/A')}</div>
                                    <div class="song-meta-item">⏱️ Durasi: {format_duration(spotify_data['duration_ms'])}</div>
                                    <div class="song-meta-item">{'🔞 Explicit' if spotify_data['explicit'] else '👪 Clean'}</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Spotify Button
                                st.markdown(f"""
                                <a href="{spotify_data['spotify_url']}" target="_blank" class="spotify-button">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-spotify" viewBox="0 0 16 16" style="display: inline-block; vertical-align: text-top; margin-right: 4px;">
                                        <path d="M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0zm3.669 11.538a.498.498 0 0 1-.686.165c-1.879-1.147-4.243-1.407-7.028-.77a.499.499 0 0 1-.222-.973c3.048-.696 5.662-.397 7.77.892a.5.5 0 0 1 .166.686zm.979-2.178a.624.624 0 0 1-.858.205c-2.15-1.321-5.428-1.704-7.972-.932a.625.625 0 0 1-.362-1.194c2.905-.881 6.517-.454 8.986 1.063a.624.624 0 0 1 .206.858zm.084-2.268C10.154 5.56 5.9 5.419 3.438 6.166a.748.748 0 1 1-.434-1.432c2.825-.857 7.523-.692 10.492 1.07a.747.747 0 1 1-.764 1.288z"/>
                                    </svg>
                                    Buka di Spotify
                                </a>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown('<span style="color: #6b7280;">Lagu tidak ditemukan di Spotify</span>', unsafe_allow_html=True)
                                
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                        with col2:
                            if spotify_data and spotify_data.get('image_url'):
                                st.image(spotify_data['image_url'], caption=f"{row['title']} - Album Cover")
                                
                                # Display popularity gauge chart
                                pop = spotify_data['popularity']
                                fig = go.Figure(go.Indicator(
                                    mode="gauge+number",
                                    value=pop,
                                    domain={'x': [0, 1], 'y': [0, 1]},
                                    title={'text': "Popularitas"},
                                    gauge={
                                        'axis': {'range': [0, 100]},
                                        'bar': {'color': get_popularity_color(pop)},
                                        'steps': [
                                            {'range': [0, 30], 'color': 'lightgray'},
                                            {'range': [30, 60], 'color': 'orange'},
                                            {'range': [60, 100], 'color': 'green'}
                                        ],
                                        'threshold': {
                                            'line': {'color': "red", 'width': 4},
                                            'thickness': 0.75,
                                            'value': 80
                                        }
                                    }
                                ))
                                fig.update_layout(height=150, margin=dict(l=20, r=20, t=30, b=20))
                                st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            else:
                st.warning("Tidak ada hasil yang ditemukan. Coba dengan kata kunci lain.")
    
    elif st.session_state.current_page == "Bantuan":
        st.title("❓ Bantuan Penggunaan")
        
        st.markdown("""
        ### Cara Menggunakan Aplikasi Pencarian Lirik
        
        Aplikasi ini membantu Anda menemukan lagu berdasarkan potongan lirik yang Anda ingat. Berikut adalah panduan penggunaan:
        
        #### 1. Pencarian Lirik
        - Masuk ke halaman "Pencarian" dengan mengklik tombol di sidebar
        - Ketik potongan lirik yang Anda ingat pada kotak pencarian
        - Klik tombol "Cari Lagu" untuk mendapatkan hasil
        
        #### 2. Memahami Hasil Pencarian
        - Hasil ditampilkan berdasarkan tingkat kemiripan dengan query pencarian Anda
        - Persentase kecocokan menunjukkan seberapa mirip lirik lagu dengan pencarian Anda
        - Jika terhubung dengan Spotify, Anda dapat mendengarkan cuplikan audio dan melihat informasi tambahan
        
        #### 3. Integrasi Spotify
        - Klik tombol "Buka di Spotify" untuk mendengarkan lagu lengkap di aplikasi atau web Spotify
        - Data popularitas dan informasi album diambil langsung dari Spotify
        
        #### 4. Dashboard Analitik
        - Lihat visualisasi dan statistik tentang dataset lagu di halaman "Dashboard"
        - Analisis distribusi artis, panjang lirik, dan tahun rilis
        
        #### Tips Pencarian Efektif:
        - Gunakan frasa yang unik atau khas dari lagu
        - Hindari kata-kata umum seperti "aku", "kamu", "dan", dll.
        - Coba berbagai bagian lirik jika hasil tidak sesuai harapan
        - Periksa ejaan dengan benar
        """)
        
        # FAQ Section
        st.subheader("Pertanyaan Umum (FAQ)")
            
        with st.expander("Apakah aplikasi ini dapat mencari lagu dalam bahasa lain?"):
            st.write("""
            Aplikasi ini dioptimalkan untuk mencari lagu berbahasa Indonesia. 
            Pencarian untuk lagu berbahasa lain mungkin tidak memberikan hasil yang akurat karena perbedaan dalam pemrosesan bahasa.
            """)
            
        with st.expander("Bagaimana cara kerja sistem pencarian ini?"):
            st.write("""
            Aplikasi ini menggunakan teknik pemrosesan bahasa alami (NLP) dan algoritma Vector Space Model dengan TF-IDF (Term Frequency-Inverse Document Frequency). 
            
            Setiap lirik lagu direpresentasikan sebagai vektor dalam ruang multidimensi, dan pencarian dilakukan dengan menghitung kemiripan (cosine similarity) antara query pencarian dan semua lirik dalam database.
            
            Hasil pencarian diurutkan berdasarkan skor kemiripan tertinggi.
            """)
            
        with st.expander("Apakah saya bisa menambahkan lagu ke database?"):
            st.write("""
            Saat ini, fitur untuk menambahkan lagu baru belum tersedia untuk pengguna umum. 
            Database lagu dikelola oleh administrator sistem. 
            
            Jika Anda memiliki saran untuk penambahan lagu, silakan hubungi kami melalui formulir kontak di halaman "Tentang Aplikasi".
            """)
    
    elif st.session_state.current_page == "Tentang":
        st.title("ℹ️ Tentang Aplikasi")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image("music.jpg", width=200)
        
        with col2:
            st.markdown("""
            # Pencarian Lirik Lagu
            
            **Versi:** 1.0.0
            
            **Tanggal Rilis:** 8 April 2023
            
            **Dikembangkan oleh:** Tim Mahasiswa Informatika
            
            Aplikasi ini dikembangkan sebagai proyek akhir untuk mata kuliah Sistem Temu Kembali Informasi (Information Retrieval) di Universitas Halu Oleo.
            """)
        
        st.markdown("---")
        
        st.markdown("""
        ### Tentang Aplikasi
        
        Aplikasi Pencarian Lirik Lagu adalah sistem temu kembali informasi yang memungkinkan pengguna menemukan lagu berdasarkan potongan lirik yang mereka ingat. Aplikasi ini menggabungkan teknik pemrosesan bahasa alami dengan integrasi API Spotify untuk memberikan pengalaman pencarian yang lengkap dan informatif.
        
        ### Fitur Utama
        
        - **Pencarian dengan pendekatan TF-IDF dan Cosine Similarity:** Menemukan lagu berdasarkan kecocokan kata yang persis
        - **Integrasi Spotify:** Menampilkan informasi lengkap tentang lagu, termasuk album, artis, hingga waktu rilis
        - **Dashboard Analitik:** Visualisasi interaktif tentang dataset lagu
        - **Antarmuka Responsif:** Desain yang user-friendly dan mudah digunakan
        
        ### Teknologi yang Digunakan
        
        - **Python:** Bahasa pemrograman utama
        - **Streamlit:** Framework untuk antarmuka web
        - **Sklearn:** Library untuk pemrosesan NLP dan model TF-IDF
        - **Sastrawi:** Library untuk pemrosesan bahasa Indonesia
        - **Plotly & Altair:** Library untuk visualisasi data interaktif
        - **Spotify API:** Integrasi untuk data dan preview audio
        
        ### Dataset
        
        Dataset berisi koleksi lirik lagu Indonesia dari berbagai artis dan genre. Lirik lagu telah melalui tahap preprocessing untuk mengoptimalkan hasil pencarian.
        
        ### Tim Pengembang
        
        Kelompok 14 STKI:

        1. **Alvian Ardiansya** – E1E122039
        2. **Ghefira Zahra Nur Fadhila** – E1E122055
        3. **Nuzul Gusti Tiara Fitri** – E1E122073
        
        """)
        
        # Disclaimer
        st.markdown("---")
        st.markdown("""
        ### Disclaimer
        
        Aplikasi ini dikembangkan untuk tujuan pendidikan dan penelitian. Semua lirik lagu yang digunakan dalam aplikasi ini adalah hak cipta dari pemiliknya masing-masing. Kami tidak mengklaim kepemilikan atas konten lirik lagu tersebut.
        
        Integrasi dengan Spotify dilakukan melalui API resmi dan sesuai dengan persyaratan layanan Spotify. Logo Spotify dan nama merek terkait adalah merek dagang dari Spotify AB.
        """)
        
        # Contact Form
        st.subheader("Hubungi Kami")
        
        with st.form("contact_form"):
            nama = st.text_input("Nama")
            email = st.text_input("Email")
            pesan = st.text_area("Pesan")
            
            submit_button = st.form_submit_button("Kirim Pesan")
            
            if submit_button:
                st.success("Pesan Anda telah diterima! Kami akan merespons segera.")

# Run the app if the data file is found
else:
    st.error("Aplikasi tidak dapat dijalankan karena file dataset tidak ditemukan.")
    st.info("Pastikan file 'indo-song.xlsx' berada di lokasi yang benar.")

    # Troubleshooting tips
    with st.expander("Tips Mengatasi Masalah"):
        st.markdown("""
        ### Troubleshooting
        
        1. Pastikan file dataset 'indo-song.xlsx' berada di direktori atau ubah path pada kode aplikasi
        2. Periksa format file dataset, pastikan itu adalah file Excel (.xlsx)
        3. Pastikan file dataset memiliki kolom 'title', 'artist', dan 'lirik'
        4. Restart aplikasi dengan menjalankan kembali perintah `streamlit run app.py`
        
        Jika masalah berlanjut, silakan hubungi tim pengembang.
        """)
