import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from load_config import load_config

from sentence_transformers import SentenceTransformer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Movie Recommendation Engine",
    page_icon="🎬",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f5f7fb;
    }

    .title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .movie-card {
        background-color: white;
        padding: 18px;
        border-radius: 12px;
        margin-bottom: 12px;
        box-shadow: 0px 3px 10px rgba(0,0,0,0.08);
    }

    .movie-title {
        font-size: 19px;
        font-weight: 600;
    }

    .similarity {
        font-size: 14px;
        margin-top: 6px;
    }

    .method-header {
        font-size: 23px;
        font-weight: 700;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    config = load_config()
    
    file_path3 = config['data']['file3_path']

    df = pd.read_csv(file_path3)
    

    required_columns = [
        "title",
        "overview",
        "keywords",
        "genres",
        "production_companies",
        "cast",
        "crew"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in movies.csv: {missing_columns}"
        )

    # Handle missing values
    for col in required_columns:

        df[col] = df[col].fillna("").astype(str)

    # --------------------------------------------------------
    # CREATE TAGS
    # --------------------------------------------------------

    df["tags"] = (
        df["overview"] + " " +
        df["keywords"] + " " +
        df["genres"] + " " +
        df["production_companies"] + " " +
        df["cast"] + " " +
        df["crew"]
    )

    # Clean extra spaces
    df["tags"] = df["tags"].apply(
        lambda x: " ".join(x.split())
    )

    return df


# ============================================================
# COUNT VECTORIZER
# ============================================================

@st.cache_resource
def create_count_similarity(tags):

    vectorizer = CountVectorizer(
        stop_words="english",
        max_features=50000
    )

    count_vector = vectorizer.fit_transform(tags)

    similarity = cosine_similarity(count_vector)

    return similarity


# ============================================================
# TF-IDF
# ============================================================

@st.cache_resource
def create_tfidf_similarity(tags):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=50000
    )

    tfidf_vector = vectorizer.fit_transform(tags)

    similarity = cosine_similarity(tfidf_vector)

    return similarity


# ============================================================
# SENTENCE TRANSFORMER
# ============================================================

@st.cache_resource
def load_sentence_transformer():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model


@st.cache_resource
def create_sentence_transformer_similarity(tags):

    model = load_sentence_transformer()

    embeddings = model.encode(
        list(tags),
        show_progress_bar=True,
        batch_size=32
    )

    similarity = cosine_similarity(
        embeddings
    )

    return similarity


# ============================================================
# RECOMMENDATION FUNCTION
# ============================================================

def get_recommendations(
    movie,
    similarity_matrix,
    df,
    number_of_recommendations=5
):

    movie_indices = df[
        df["title"].str.lower() == movie.lower()
    ].index

    if len(movie_indices) == 0:
        return []

    movie_index = movie_indices[0]

    distances = similarity_matrix[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )

    recommendations = []

    for index, score in movie_list:

        # Skip selected movie itself
        if index == movie_index:
            continue

        recommendations.append(
            {
                "Movie": df.iloc[index]["title"],
                "Similarity": round(float(score), 3)
            }
        )

        if len(recommendations) == number_of_recommendations:
            break

    return recommendations


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:

    df = load_data()

except Exception as e:

    st.error(f"Error loading dataset: {e}")
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">🎬 Movie Recommendation Engine</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Compare recommendations using Traditional NLP and
    Transformer-based Semantic Search
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Recommendation Settings")

number_of_recommendations = st.sidebar.slider(
    "Number of Recommendations",
    min_value=3,
    max_value=10,
    value=5
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Models**

    📊 CountVectorizer

    🔍 TF-IDF

    🤖 Sentence Transformer

    Model:
    `all-MiniLM-L6-v2`
    """
)


# ============================================================
# MOVIE SELECTION
# ============================================================

movie_list = sorted(
    df["title"].dropna().unique().tolist()
)

selected_movie = st.selectbox(
    "🎥 Select a Movie",
    movie_list
)


# ============================================================
# BUTTON
# ============================================================

recommend_button = st.button(
    "🚀 Generate Recommendations",
    use_container_width=True
)


# ============================================================
# RECOMMENDATIONS
# ============================================================

if recommend_button:

    # --------------------------------------------------------
    # CREATE MODELS
    # --------------------------------------------------------

    with st.spinner("Preparing recommendation models..."):

        count_similarity = create_count_similarity(
            df["tags"]
        )

        tfidf_similarity = create_tfidf_similarity(
            df["tags"]
        )

        sentence_similarity = (
            create_sentence_transformer_similarity(
                df["tags"]
            )
        )


    # --------------------------------------------------------
    # GET RECOMMENDATIONS
    # --------------------------------------------------------

    count_results = get_recommendations(
        selected_movie,
        count_similarity,
        df,
        number_of_recommendations
    )

    tfidf_results = get_recommendations(
        selected_movie,
        tfidf_similarity,
        df,
        number_of_recommendations
    )

    sentence_results = get_recommendations(
        selected_movie,
        sentence_similarity,
        df,
        number_of_recommendations
    )


    # --------------------------------------------------------
    # SELECTED MOVIE
    # --------------------------------------------------------

    st.success(
        f"Recommendations generated for **{selected_movie}**"
    )

    st.markdown("---")


    # ========================================================
    # THREE COLUMNS
    # ========================================================

    col1, col2, col3 = st.columns(3)


    # ========================================================
    # COUNT VECTORIZER
    # ========================================================

    with col1:

        st.markdown(
            '<div class="method-header">'
            '📊 CountVectorizer'
            '</div>',
            unsafe_allow_html=True
        )

        st.caption(
            "Bag-of-Words based similarity"
        )

        for i, result in enumerate(count_results, 1):

            st.markdown(
                f"""
                <div class="movie-card">

                    <div class="movie-title">
                        {i}. {result["Movie"]}
                    </div>

                    <div class="similarity">
                        Similarity:
                        <b>{result["Similarity"]}</b>
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


    # ========================================================
    # TF-IDF
    # ========================================================

    with col2:

        st.markdown(
            '<div class="method-header">'
            '🔍 TF-IDF'
            '</div>',
            unsafe_allow_html=True
        )

        st.caption(
            "Term-frequency based similarity"
        )

        for i, result in enumerate(tfidf_results, 1):

            st.markdown(
                f"""
                <div class="movie-card">

                    <div class="movie-title">
                        {i}. {result["Movie"]}
                    </div>

                    <div class="similarity">
                        Similarity:
                        <b>{result["Similarity"]}</b>
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


    # ========================================================
    # SENTENCE TRANSFORMER
    # ========================================================

    with col3:

        st.markdown(
            '<div class="method-header">'
            '🤖 Sentence Transformer'
            '</div>',
            unsafe_allow_html=True
        )

        st.caption(
            "Semantic embedding based similarity"
        )

        for i, result in enumerate(sentence_results, 1):

            st.markdown(
                f"""
                <div class="movie-card">

                    <div class="movie-title">
                        {i}. {result["Movie"]}
                    </div>

                    <div class="similarity">
                        Similarity:
                        <b>{result["Similarity"]}</b>
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


    # ========================================================
    # COMPARISON
    # ========================================================

    st.markdown("---")

    st.subheader("📈 Recommendation Comparison")

    comparison_data = []

    for i in range(number_of_recommendations):

        row = {
            "Rank": i + 1,
            "CountVectorizer":
                count_results[i]["Movie"]
                if i < len(count_results)
                else "",

            "Count Score":
                count_results[i]["Similarity"]
                if i < len(count_results)
                else 0,

            "TF-IDF":
                tfidf_results[i]["Movie"]
                if i < len(tfidf_results)
                else "",

            "TF-IDF Score":
                tfidf_results[i]["Similarity"]
                if i < len(tfidf_results)
                else 0,

            "Sentence Transformer":
                sentence_results[i]["Movie"]
                if i < len(sentence_results)
                else "",

            "Transformer Score":
                sentence_results[i]["Similarity"]
                if i < len(sentence_results)
                else 0
        }

        comparison_data.append(row)


    comparison_df = pd.DataFrame(
        comparison_data
    )

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Movie Recommendation Engine | "
    "CountVectorizer + TF-IDF + Sentence Transformers + "
    "Cosine Similarity"
)