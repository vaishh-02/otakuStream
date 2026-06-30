# OtakuStream

OtakuStream is a Python project that recommends anime based on content similarity and helps users build a watch plan within a time limit. It combines natural language processing with a simple greedy algorithm to make the experience more practical.

## Key Features
- Content-based filtering using Scikit-Learn's TF-IDF vectorizer and cosine similarity.
- A training pipeline that prepares the data and saves the model artifacts.
- A recommender script that loads those saved files and generates recommendations on demand.
- A binge-watch planner that selects anime by rating while staying within a user-defined hour budget.

## How It Works
### 1. Text-Based Recommendation
The system combines each anime's genre and synopsis into a single text description. It then converts that text into numerical vectors using TF-IDF. Similar anime are found by comparing those vectors.

### 2. Memory-Friendly Design
Instead of storing a huge similarity matrix for every anime pair, the project saves the cleaned data, the fitted vectorizer, and the sparse TF-IDF matrix. Recommendations are computed when needed, which reduces memory usage.

### 3. Greedy Watch Planning
The binge-watch planner sorts anime by rating and adds them to a watchlist until the total estimated watch time reaches the user's limit. This is a greedy approach that is simple and fast.

## Project Structure
```text
OtakuStream/
├── train_model.py
├── recommender.py
├── .gitignore
└── README.md
```