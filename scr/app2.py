"""
Production-ready Streamlit Movie Recommendation Engine.

Methods:
    1. CountVectorizer + Cosine Similarity
    2. TF-IDF + Cosine Similarity
    3. Sentence Transformer + Cosine Similarity

Expected load_config() structure:

{
    "data": {
        "file1_path": "path/to/movies.csv",
        "file2_path": "path/to/credits.csv"
    }
}
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
)

from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

from load_config import load_config


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Movie Recommendation Engine",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# APPLICATION CONSTANTS
# ============================================================

REQUIRED_COLUMNS = [
    "title",
    "overview",
    "keywords",
    "genres",
    "production_companies",
    "cast",
    "crew",
]

MODEL_NAME = "all-MiniLM-L6-v2"

MAX_FEATURES = 50_000


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .app-subtitle {
            text-align: center;
            color: #6b7280;
            margin-top: -0.5rem;
            margin-bottom: 2rem;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.15rem;
        }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """
    Load and prepare movie data.

    Returns:
        Prepared pandas DataFrame.

    Raises:
        ValueError:
            If configuration or required columns are missing.
    """

    config = load_config()

    # --------------------------------------------------------
    # Read configuration
    # --------------------------------------------------------

    try:
        movies_path = config["data"]["file1_path"]
        credits_path = config["data"]["file2_path"]

    except KeyError as exc:

        raise ValueError(
            "Configuration must contain "
            "'data.file1_path' and 'data.file2_path'."
        ) from exc

    # --------------------------------------------------------
    # Read CSV files
    # --------------------------------------------------------

    movies = pd.read_csv(movies_path)

    credits = pd.read_csv(credits_path)

    # --------------------------------------------------------
    # Validate title column
    # --------------------------------------------------------

    if "title" not in movies.columns:

        raise ValueError(
            "Movies dataset must contain a 'title' column."
        )

    if "title" not in credits.columns:

        raise ValueError(
            "Credits dataset must contain a 'title' column."
        )

    # --------------------------------------------------------
    # Merge datasets
    # --------------------------------------------------------

    df = credits.merge(
        movies,
        on="title",
        how="inner",
    )

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns after merge: "
            + ", ".join(missing_columns)
        )

    # --------------------------------------------------------
    # Clean data
    # --------------------------------------------------------

    df = df.copy()

    for column in REQUIRED_COLUMNS:

        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    # Remove records without movie titles

    df = df[
        df["title"].ne("")
    ].reset_index(drop=True)

    # --------------------------------------------------------
    # Create combined tags
    # --------------------------------------------------------

    df["tags"] = (
        df["overview"]
        + " "
        + df["keywords"]
        + " "
        + df["genres"]
        + " "
        + df["production_companies"]
        + " "
        + df["cast"]
        + " "
        + df["crew"]
    )

    # Remove duplicate whitespace

    df["tags"] = df["tags"].apply(
        lambda value: " ".join(value.split())
    )

    return df


# ============================================================
# COUNT VECTORIZER
# ============================================================

@st.cache_resource(show_spinner=False)
def create_count_vectorizer(
    tags: tuple[str, ...],
):
    """
    Create and cache CountVectorizer matrix.
    """

    vectorizer = CountVectorizer(
        stop_words="english",
        max_features=MAX_FEATURES,
    )

    matrix = vectorizer.fit_transform(tags)

    return matrix


# ============================================================
# TF-IDF
# ============================================================

@st.cache_resource(show_spinner=False)
def create_tfidf_vectorizer(
    tags: tuple[str, ...],
):
    """
    Create and cache TF-IDF matrix.
    """

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=MAX_FEATURES,
    )

    matrix = vectorizer.fit_transform(tags)

    return matrix


# ============================================================
# SENTENCE TRANSFORMER
# ============================================================

@st.cache_resource(show_spinner=False)
def load_sentence_transformer():
    """
    Load and cache Sentence Transformer model.
    """

    model = SentenceTransformer(
        MODEL_NAME
    )

    return model


@st.cache_resource(show_spinner=False)
def create_sentence_embeddings(
    tags: tuple[str, ...],
):
    """
    Generate and cache sentence embeddings.
    """

    model = load_sentence_transformer()

    embeddings = model.encode(
        list(tags),
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings


# ============================================================
# MOVIE LOOKUP
# ============================================================

def find_movie_index(
    df: pd.DataFrame,
    movie: str,
):
    """
    Find the index of a movie using case-insensitive matching.
    """

    matches = df.index[
        df["title"]
        .str.casefold()
        .eq(movie.casefold())
    ]

    if len(matches) == 0:

        return None

    return int(matches[0])


# ============================================================
# BUILD RECOMMENDATIONS
# ============================================================

def build_recommendation_results(
    df: pd.DataFrame,
    movie_index: int,
    scores: np.ndarray,
    number_of_recommendations: int,
) -> List[Dict[str, object]]:
    """
    Convert similarity scores into ranked recommendations.
    """

    # We only need the top N + the selected movie.
    candidate_count = min(
        len(scores),
        number_of_recommendations + 1,
    )

    # Get indexes of highest scores.
    candidate_indices = np.argpartition(
        scores,
        -candidate_count,
    )[-candidate_count:]

    # Sort candidates from highest to lowest.
    candidate_indices = candidate_indices[
        np.argsort(
            scores[candidate_indices]
        )[::-1]
    ]

    recommendations = []

    for index in candidate_indices:

        index = int(index)

        # Never recommend the selected movie.
        if index == movie_index:
            continue

        recommendations.append(
            {
                "Movie": str(
                    df.iloc[index]["title"]
                ),
                "Similarity": round(
                    float(scores[index]),
                    3,
                ),
            }
        )

        if (
            len(recommendations)
            >= number_of_recommendations
        ):
            break

    return recommendations


# ============================================================
# SPARSE MODEL RECOMMENDATIONS
# ============================================================

def get_sparse_recommendations(
    movie: str,
    matrix,
    df: pd.DataFrame,
    number_of_recommendations: int,
):
    """
    Generate recommendations for CountVectorizer
    or TF-IDF.

    Important:
    We calculate similarity only for the selected movie
    instead of creating an entire N x N similarity matrix.
    """

    movie_index = find_movie_index(
        df,
        movie,
    )

    if movie_index is None:

        return []

    # Get selected movie vector.

    selected_vector = matrix[
        movie_index
    ]

    # Compare selected movie against all movies.

    scores = cosine_similarity(
        selected_vector,
        matrix,
    ).ravel()

    return build_recommendation_results(
        df,
        movie_index,
        scores,
        number_of_recommendations,
    )


# ============================================================
# TRANSFORMER RECOMMENDATIONS
# ============================================================

def get_transformer_recommendations(
    movie: str,
    embeddings: np.ndarray,
    df: pd.DataFrame,
    number_of_recommendations: int,
):
    """
    Generate recommendations using Sentence Transformer.

    Because embeddings are normalized, cosine similarity
    can be calculated using a dot product.
    """

    movie_index = find_movie_index(
        df,
        movie,
    )

    if movie_index is None:

        return []

    selected_embedding = embeddings[
        movie_index
    ]

    scores = (
        embeddings
        @ selected_embedding
    )

    return build_recommendation_results(
        df,
        movie_index,
        scores,
        number_of_recommendations,
    )


# ============================================================
# RECOMMENDATION UI
# ============================================================

def render_recommendations(
    results: List[Dict[str, object]],
    title: str,
    description: str,
):
    """
    Render recommendation results using native
    Streamlit components.
    """

    st.subheader(title)

    st.caption(description)

    if not results:

        st.info(
            "No recommendations found."
        )

        return

    for rank, result in enumerate(
        results,
        start=1,
    ):

        movie_name = str(
            result["Movie"]
        )

        similarity = float(
            result["Similarity"]
        )

        # Native Streamlit container.
        # No raw movie HTML required.

        with st.container(
            border=True
        ):

            movie_column, score_column = (
                st.columns(
                    [4, 1]
                )
            )

            with movie_column:

                st.markdown(
                    f"**{rank}. {movie_name}**"
                )

            with score_column:

                st.metric(
                    label="Similarity",
                    value=f"{similarity:.3f}",
                )


# ============================================================
# COMPARISON TABLE
# ============================================================

def build_comparison_dataframe(
    count_results,
    tfidf_results,
    transformer_results,
    number_of_recommendations,
):
    """
    Build comparison DataFrame.
    """

    rows = []

    for rank in range(
        number_of_recommendations
    ):

        # CountVectorizer result

        if rank < len(count_results):

            count_movie = (
                count_results[rank]["Movie"]
            )

            count_score = (
                count_results[rank]["Similarity"]
            )

        else:

            count_movie = "—"

            count_score = 0.0

        # TF-IDF result

        if rank < len(tfidf_results):

            tfidf_movie = (
                tfidf_results[rank]["Movie"]
            )

            tfidf_score = (
                tfidf_results[rank]["Similarity"]
            )

        else:

            tfidf_movie = "—"

            tfidf_score = 0.0

        # Transformer result

        if rank < len(
            transformer_results
        ):

            transformer_movie = (
                transformer_results[
                    rank
                ]["Movie"]
            )

            transformer_score = (
                transformer_results[
                    rank
                ]["Similarity"]
            )

        else:

            transformer_movie = "—"

            transformer_score = 0.0

        rows.append(
            {
                "Rank": rank + 1,

                "CountVectorizer":
                    count_movie,

                "Count Score":
                    count_score,

                "TF-IDF":
                    tfidf_movie,

                "TF-IDF Score":
                    tfidf_score,

                "Sentence Transformer":
                    transformer_movie,

                "Transformer Score":
                    transformer_score,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# SIDEBAR MODEL INFORMATION
# ============================================================

def render_model_information():
    """
    Display model information in sidebar.
    """

    with st.expander(
        "🤖 Models",
        expanded=False,
    ):

        st.markdown(
            """
            **Traditional NLP**

            - CountVectorizer
            - TF-IDF

            **Semantic NLP**

            - Sentence Transformer
            - `all-MiniLM-L6-v2`

            **Similarity**

            - Cosine Similarity
            """
        )


# ============================================================
# LOAD DATA
# ============================================================

try:

    df = load_data()

except Exception as exc:

    st.error(
        "❌ Unable to load the movie dataset."
    )

    st.exception(exc)

    st.stop()


# ============================================================
# EMPTY DATASET CHECK
# ============================================================

if df.empty:

    st.error(
        "❌ The dataset contains no usable movie records."
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title(
    "🎬 Movie Recommendation Engine"
)

st.markdown(
    """
    <div class="app-subtitle">

    Compare movie recommendations using
    traditional NLP and transformer-based
    semantic search.

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Recommendation Settings"
    )

    number_of_recommendations = (
        st.slider(
            "Number of Recommendations",
            min_value=3,
            max_value=10,
            value=5,
        )
    )

    st.divider()

    render_model_information()

    st.caption(
        f"Dataset: {len(df):,} movies"
    )


# ============================================================
# MOVIE LIST
# ============================================================

movie_list = sorted(
    df["title"]
    .drop_duplicates()
    .tolist(),
    key=str.casefold,
)


# ============================================================
# MOVIE SELECTOR
# ============================================================

selected_movie = st.selectbox(
    "🎥 Select a Movie",
    options=movie_list,
)


# ============================================================
# GENERATE BUTTON
# ============================================================

generate_button = st.button(
    "🚀 Generate Recommendations",
    type="primary",
    use_container_width=True,
)


# ============================================================
# GENERATE RECOMMENDATIONS
# ============================================================

if generate_button:

    # --------------------------------------------------------
    # Convert tags to tuple so Streamlit can cache it.
    # --------------------------------------------------------

    tags = tuple(
        df["tags"].tolist()
    )

    # --------------------------------------------------------
    # Load / create models
    # --------------------------------------------------------

    try:

        with st.status(
            "Preparing recommendation models...",
            expanded=True,
        ) as status:

            st.write(
                "Preparing CountVectorizer..."
            )

            count_matrix = (
                create_count_vectorizer(
                    tags
                )
            )

            st.write(
                "Preparing TF-IDF..."
            )

            tfidf_matrix = (
                create_tfidf_vectorizer(
                    tags
                )
            )

            st.write(
                "Preparing Sentence Transformer..."
            )

            transformer_embeddings = (
                create_sentence_embeddings(
                    tags
                )
            )

            status.update(
                label=(
                    "Recommendation models ready."
                ),
                state="complete",
                expanded=False,
            )

    except Exception as exc:

        st.error(
            "❌ The recommendation models "
            "could not be prepared."
        )

        st.exception(exc)

        st.stop()

    # --------------------------------------------------------
    # Generate recommendations
    # --------------------------------------------------------

    with st.spinner(
        "Finding similar movies..."
    ):

        count_results = (
            get_sparse_recommendations(
                selected_movie,
                count_matrix,
                df,
                number_of_recommendations,
            )
        )

        tfidf_results = (
            get_sparse_recommendations(
                selected_movie,
                tfidf_matrix,
                df,
                number_of_recommendations,
            )
        )

        transformer_results = (
            get_transformer_recommendations(
                selected_movie,
                transformer_embeddings,
                df,
                number_of_recommendations,
            )
        )

    # --------------------------------------------------------
    # Success message
    # --------------------------------------------------------

    st.success(
        f"Recommendations generated for "
        f"**{selected_movie}**"
    )

    st.divider()

    # ========================================================
    # THREE RECOMMENDATION METHODS
    # ========================================================

    col1, col2, col3 = st.columns(3)

    # --------------------------------------------------------
    # CountVectorizer
    # --------------------------------------------------------

    with col1:

        render_recommendations(
            results=count_results,
            title="📊 CountVectorizer",
            description=(
                "Bag-of-Words based similarity"
            ),
        )

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    with col2:

        render_recommendations(
            results=tfidf_results,
            title="🔍 TF-IDF",
            description=(
                "Term-frequency based similarity"
            ),
        )

    # --------------------------------------------------------
    # Sentence Transformer
    # --------------------------------------------------------

    with col3:

        render_recommendations(
            results=transformer_results,
            title="🤖 Sentence Transformer",
            description=(
                "Semantic embedding similarity"
            ),
        )

    # ========================================================
    # COMPARISON
    # ========================================================

    st.divider()

    st.subheader(
        "📈 Recommendation Comparison"
    )

    comparison_df = (
        build_comparison_dataframe(
            count_results=count_results,
            tfidf_results=tfidf_results,
            transformer_results=(
                transformer_results
            ),
            number_of_recommendations=(
                number_of_recommendations
            ),
        )
    )

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
        height=min(
            500,
            70
            + number_of_recommendations * 40,
        ),
        column_config={

            "Rank":
                st.column_config.NumberColumn(
                    "Rank",
                    width="small",
                ),

            "Count Score":
                st.column_config.NumberColumn(
                    "Count Score",
                    format="%.3f",
                ),

            "TF-IDF Score":
                st.column_config.NumberColumn(
                    "TF-IDF Score",
                    format="%.3f",
                ),

            "Transformer Score":
                st.column_config.NumberColumn(
                    "Transformer Score",
                    format="%.3f",
                ),
        },
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Movie Recommendation Engine • "
    "CountVectorizer + TF-IDF + "
    "Sentence Transformers + "
    "Cosine Similarity"
)