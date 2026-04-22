import streamlit as st
import pickle
import pandas as pd
import ast
import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="centered"
)

# ─── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0d0d;
    color: #f0f0f0;
}

h1 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4rem !important;
    letter-spacing: 4px;
    color: #e50914;
    margin-bottom: 0 !important;
}

.subtitle {
    color: #888;
    font-size: 0.9rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

.stSelectbox > div > div {
    background-color: #1a1a1a !important;
    border: 1px solid #333 !important;
    border-radius: 4px !important;
    color: #f0f0f0 !important;
}

.stButton > button {
    background-color: #e50914 !important;
    color: white !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.2rem !important;
    letter-spacing: 2px !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 0.5rem 2rem !important;
    width: 100% !important;
    transition: background 0.2s !important;
}

.stButton > button:hover {
    background-color: #b0060f !important;
}

.movie-card {
    background: #1a1a1a;
    border-left: 3px solid #e50914;
    border-radius: 4px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 1rem;
    font-weight: 500;
    color: #f0f0f0;
    letter-spacing: 0.5px;
}

.rank {
    color: #e50914;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.2rem;
    margin-right: 0.5rem;
}

.divider {
    border: none;
    border-top: 1px solid #222;
    margin: 1.5rem 0;
}

footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─── Load & Build Model ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    nltk.download('punkt', quiet=True)
    ps = PorterStemmer()

    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')

    movies = movies.merge(credits, on='title')
    movies = movies[['movie_id', 'title', 'genres', 'keywords', 'overview', 'cast', 'crew']]
    movies = movies.dropna()

    def convert(obj):
        return [i['name'] for i in ast.literal_eval(obj)]

    def convert3(obj):
        return [i['name'] for idx, i in enumerate(ast.literal_eval(obj)) if idx < 3]

    def fetch_director(obj):
        return [i['name'] for i in ast.literal_eval(obj) if i['job'] == 'Director']

    def stem(text):
        return " ".join([ps.stem(w) for w in text.split()])

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert3)
    movies['crew'] = movies['crew'].apply(fetch_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())

    for col in ['crew', 'overview', 'genres', 'cast', 'keywords']:
        movies[col] = movies[col].apply(lambda x: [i.replace(' ', '') for i in x])

    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['crew'] + movies['cast']

    new_df = movies[['movie_id', 'title', 'tags']].copy()
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())
    new_df['tags'] = new_df['tags'].apply(stem)

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    similarity = cosine_similarity(vectors)

    return new_df, similarity

# ─── Recommend Function ───────────────────────────────────────────────────────
def recommend(movie, new_df, similarity):
    idx = new_df[new_df['title'] == movie].index[0]
    distances = similarity[idx]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [new_df.iloc[i[0]].title for i in movies_list]

# ─── UI ───────────────────────────────────────────────────────────────────────
st.markdown('<h1>CineMatch</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Content-Based Movie Recommendation</p>', unsafe_allow_html=True)

with st.spinner("Loading model..."):
    try:
        new_df, similarity = load_model()
        model_loaded = True
    except Exception as e:
        model_loaded = False
        st.error(f"⚠️ Could not load CSV files. Make sure `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` are in the same folder.\n\n`{e}`")

if model_loaded:
    movie_list = new_df['title'].values
    selected_movie = st.selectbox("🎬 Search or select a movie", movie_list)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    if st.button("GET RECOMMENDATIONS"):
        results = recommend(selected_movie, new_df, similarity)
        st.markdown("### You might also like:")
        for rank, title in enumerate(results, 1):
            st.markdown(
                f'<div class="movie-card"><span class="rank">#{rank}</span>{title}</div>',
                unsafe_allow_html=True
            )