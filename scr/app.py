import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from load_config import load_config


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    color: #4B0082;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #666666;
    margin-bottom: 30px;
}

.movie-card {
    background: white;
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 12px;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.08);
}

.movie-title {
    font-size: 20px;
    font-weight: 600;
    color: #333333;
}

.similarity {
    font-size: 15px;
    color: #666666;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

@st.cache_data
def load_data():

    config = load_config()

    file_path1 = config['data']['file1_path']
    file_path2 = config['data']['file2_path']

    df_mv = pd.read_csv(file_path1)
    df_cr = pd.read_csv(file_path2)

    df=df_cr.merge(df_mv,on='title')

    # Fill missing values
    columns = [
        "overview",
        "keywords",
        "genres",
        "production_companies",
        "cast",
        "crew"
    ]

    for col in columns:
        df[col] = df[col].fillna("")

    # Combine all information
    df["tags"] = (
        df["overview"] + " " +
        df["keywords"] + " " +
        df["genres"] + " " +
        df["production_companies"] + " " +
        df["cast"] + " " +
        df["crew"]
    )

    # Clean tags
    df["tags"] = df["tags"].apply(
        lambda x: " ".join(str(x).split())
    )

    return df


df = load_data()


# ---------------------------------------------------------
# COUNT VECTORIZER
# ---------------------------------------------------------

@st.cache_resource
def create_count_vectorizer(df):

    vectorizer = CountVectorizer(
        stop_words="english",
        max_features=50000
    )

    count_vector = vectorizer.fit_transform(
        df["tags"]
    )

    similarity = cosine_similarity(count_vector)

    return similarity


# ---------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------

@st.cache_resource
def create_tfidf_vectorizer(df):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=50000
    )

    tfidf_vector = vectorizer.fit_transform(
        df["tags"]
    )

    similarity = cosine_similarity(tfidf_vector)

    return similarity


similar_count = create_count_vectorizer(df)
similar_tfidf = create_tfidf_vectorizer(df)


# ---------------------------------------------------------
# RECOMMENDATION FUNCTION
# ---------------------------------------------------------

def get_recommendations(movie, similarity_matrix, n=5):

    movie_index = df[
        df["title"].str.lower() == movie.lower()
    ].index

    if len(movie_index) == 0:
        return []

    movie_index = movie_index[0]

    distances = similarity_matrix[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:n+1]

    recommendations = []

    for index, score in movie_list:

        recommendations.append({
            "Movie": df.iloc[index]["title"],
            "Similarity": round(score, 3)
        })

    return recommendations


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    '<div class="title">🎬 Movie Recommendation System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Compare recommendations using CountVectorizer and TF-IDF'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header("⚙️ Recommendation Settings")

num_recommendations = st.sidebar.slider(
    "Number of Recommendations",
    min_value=5,
    max_value=10,
    value=5
)


# ---------------------------------------------------------
# MOVIE SELECTION
# ---------------------------------------------------------

movie_list = sorted(
    df["title"].dropna().unique().tolist()
)

selected_movie = st.selectbox(
    "🎥 Select a Movie",
    movie_list
)


# ---------------------------------------------------------
# RECOMMEND BUTTON
# ---------------------------------------------------------

recommend = st.button(
    "🚀 Get Recommendations",
    use_container_width=True
)


# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

if recommend:

    st.markdown(
        f"### Recommendations for **{selected_movie}**"
    )

    count_results = get_recommendations(
        selected_movie,
        similar_count,
        num_recommendations
    )

    tfidf_results = get_recommendations(
        selected_movie,
        similar_tfidf,
        num_recommendations
    )

    col1, col2 = st.columns(2)

    # -----------------------------------------------------
    # COUNT VECTOR RESULTS
    # -----------------------------------------------------

    with col1:

        st.subheader("📊 CountVectorizer")

        if count_results:

            for i, result in enumerate(count_results, 1):

                st.markdown(
                    f"""
                    <div class="movie-card">
                        <div class="movie-title">
                            {i}. {result["Movie"]}
                        </div>
                        <div class="similarity">
                            Similarity Score:
                            <b>{result["Similarity"]}</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:

            st.warning("Movie not found.")


    # -----------------------------------------------------
    # TF-IDF RESULTS
    # -----------------------------------------------------

    with col2:

        st.subheader("🔍 TF-IDF")

        if tfidf_results:

            for i, result in enumerate(tfidf_results, 1):

                st.markdown(
                    f"""
                    <div class="movie-card">
                        <div class="movie-title">
                            {i}. {result["Movie"]}
                        </div>
                        <div class="similarity">
                            Similarity Score:
                            <b>{result["Similarity"]}</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:

            st.warning("Movie not found.")


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown("---")

st.caption(
    "Built with Python • Streamlit • CountVectorizer • TF-IDF • Cosine Similarity"
)