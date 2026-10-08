import json
import logging
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from load_config import load_config

# Configure logging for production monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def safe_json_parse(json_str, key_name="name", max_items=None, filter_job=None):
    """
    Safely parses TMDB JSON strings to extract text fields.
    Handles NaN values and malformed JSON strings without crashing.
    """
    if pd.isna(json_str) or not isinstance(json_str, str):
        return []
    try:
        data = json.loads(json_str)
        if filter_job:
            return [item[key_name] for item in data if item.get('job') == filter_job]
        
        items = [item[key_name] for item in data]
        return items[:max_items] if max_items else items
    except (json.JSONDecodeError, TypeError) as e:
        logging.warning(f"Failed to parse JSON string: {e}")
        return []

def clean_spaces(item_list):
    """Removes all internal spaces within elements of a list for tight tag tokenization."""
    if not isinstance(item_list, list):
        return []
    return [str(item).replace(" ", "") for item in item_list]

def build_recommendation_system():
    """Pipeline to load data, clean features, and generate similarity matrices."""
    logging.info("Loading configuration and data files...")
    config = load_config()
    
    df_mv = pd.read_csv(config['data']['file1_path'])
    df_cr = pd.read_csv(config['data']['file2_path'])
    
    # Merge datasets on movie title
    df_new = df_cr.merge(df_mv, on='title')
    
    # Select relevant features
    df = df_new[['id', 'title', 'genres', 'keywords', 'production_companies', 'cast', 'crew', 'overview']].copy()
    
    logging.info("Parsing structured JSON columns...")
    df['genres'] = df['genres'].apply(lambda x: safe_json_parse(x))
    df['keywords'] = df['keywords'].apply(lambda x: safe_json_parse(x))
    df['production_companies'] = df['production_companies'].apply(lambda x: safe_json_parse(x))
    df['cast'] = df['cast'].apply(lambda x: safe_json_parse(x, max_items=5))
    df['crew'] = df['crew'].apply(lambda x: safe_json_parse(x, filter_job='Director'))
    
    # Process text overview into a tokenized list
    df['overview'] = df['overview'].fillna('').str.split()
    
    logging.info("Removing spaces to form unique categorical tokens...")
    columns_to_strip = ['genres', 'keywords', 'production_companies', 'cast', 'crew', 'overview']
    for col in columns_to_strip:
        df[col] = df[col].apply(clean_spaces)
        
    # Combine all structural features into a consolidated 'tags' string
    df['tags'] = df['overview'] + df['keywords'] + df['genres'] + df['production_companies'] + df['cast'] + df['crew']
    df['tags'] = df['tags'].apply(lambda x: ' '.join(x).lower())
    
    # Drop raw components to optimize memory footprint
    df = df.drop(columns=columns_to_strip)
    
    logging.info("Vectorizing features and computing matrix shapes...")
    # Compute Count Vectorizer Similarity Matrix
    cv = CountVectorizer(stop_words='english', max_features=50000)
    count_matrix = cv.fit_transform(df['tags'])
    similarity_cv = cosine_similarity(count_matrix)
    
    # Compute TF-IDF Vectorizer Similarity Matrix
    tfidf = TfidfVectorizer(stop_words='english', max_features=50000)
    tfidf_matrix = tfidf.fit_transform(df['tags'])
    similarity_tfidf = cosine_similarity(tfidf_matrix)
    
    return df, similarity_cv, similarity_tfidf

# Run pipeline execution
df, similar_ct_vector, similar_tfidf_vector = build_recommendation_system()

def recommend_movie(movie_title, similarity_matrix, method_label="System"):
    """
    Finds and prints the top 5 closest recommendations for a given movie title.
    Includes validation handling for safe production usage.
    """
    # Case-insensitive validation mapping
    match_idx = df[df['title'].str.lower() == movie_title.lower()].index
    
    if match_idx.empty:
        print(f"⚠️ Error: '{movie_title}' was not found in the movie catalog.")
        return
        
    movie_index = match_idx[0]
    distances = similarity_matrix[movie_index]
    
    # Rank top items skipping the self-comparison index
    recommendation_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    print(f"\n--- Recommendations using {method_label} for '{movie_title}' ---")
    for idx, score in recommendation_list:
        print(f"-> {df.iloc[idx].title:<40} (Match Score: {round(score, 3)})")

# Execute verification runs
recommend_movie('Avatar', similar_ct_vector, method_label="Count Vectorizer")
recommend_movie('Avatar', similar_tfidf_vector, method_label="TF-IDF")
