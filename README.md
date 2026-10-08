# 🎬 Movie Recommendation Engine

An interactive, production-ready web application built with **Streamlit** and **Scikit-Learn** that provides multi-model movie recommendations. The system leverages cross-paradigm NLP pipelines—comparing classical frequency-based methods with state-of-the-art **Transformer Embeddings** to deliver high-precision semantic matchmaking.

---

## 🚀 Key Engine Features

* **Multi-Algorithm Comparison:** Generates side-by-side performance checks evaluating three distinct text approaches on identical metadata fields:
  * 📊 **CountVectorizer:** Frequency and bag-of-words keyword pairing.
  * 🔍 **TF-IDF Vectorizer:** Scaled importance metrics penalizing globally saturated terms.
  * 🤖 **Sentence Transformers (`all-MiniLM-L6-v2`):** Multi-dimensional vector embeddings mapping deep contextual and semantic meaning.
* **Unified Component Tagging:** Synthesizes isolated variables (`overview`, `keywords`, `genres`, `production_companies`, `cast`, and `crew`) into dense structural tracking arrays.
* **Dynamic UX Settings:** Control parameters globally from a unified sidebar, adjusting recommendation boundaries dynamically between 3 and 10 results.
* **Production-Grade Infrastructure:** Utilizes Streamlit's structural caching mechanisms (`@st.cache_data` and `@st.cache_resource`) to hold vector models and weights safely in system memory for instant query responses.

---

## 📂 Project Architecture

Organize your active workspace folder using the following tree structure:

```text
├── .streamlit/
│   ├── config.toml           # Streamlit application layout settings
│   └── secrets.toml          # Encrypted platform keys (if needed)
├── data/
│   └── dataset.csv.gz        # Gzip compressed movie dataset
├── config.toml               # Environment configuration rules
├── load_config.py            # Local utility parser mapping file paths
└── app.py                    # Main Streamlit web application code
```

---

## 🛠️ Installation & Environment Setup

### 1. Extract Workspace Files
```bash
git clone https://github.com
cd movie-recommendation-engine
```

### 2. Configure Dependencies
Make sure you have Python 3.9+ deployed on your host machine, then install the structural framework packages:
```bash
pip install streamlit pandas scikit-learn sentence-transformers
```

### 3. Application File Routing Configuration
Confirm your configurations (typically inside `config.toml`) accurately match your compressed file path parameters:
```toml
[data]
file3_path = "data/dataset.csv.gz"
```

---

## 💻 Running the Application Locally

Fire up your local web server by executing the following terminal script command:
```bash
streamlit run app.py
```

### 🔄 Data Processing Lifecycle
1. **Dynamic In-Memory Pull:** `load_data()` fetches file coordinates from your config utility, loading the source directly into a working DataFrame.
2. **Metadata Sanitization:** Validates dataset structures for critical attributes, drops redundant structural formatting anomalies, cleans dangling white spaces, and aggregates text metadata features into a global `tags` column.
3. **Similarity Mapping:** Computes multidimensional metric evaluations across the entire dataset via global spatial cosine mapping.
4. **Side-by-Side Analysis:** Renders a granular summary grid benchmarking scores down to individual fractional metrics (`round(score, 3)`).

---

## 🔒 Production Best Practices & Deployment Guidelines

* **Version Control Boundaries:** Large datasets and pre-computed similarity matrix profiles should never be pushed onto public GitHub repositories. Ensure your `.gitignore` configuration tracks large files (`*.csv`, `*.pkl`) and blocks configuration logs like `.streamlit/secrets.toml`.
* **Resource Optimization:** The model utilizes heavy local memory overhead during the training phase of `all-MiniLM-L6-v2`. If deploying via **Streamlit Community Cloud**, ensure your memory limits are managed gracefully through pre-computed matrix storage (`joblib` or `pickle`) to avoid unexpected RAM usage kills.
# movie_recomendation
