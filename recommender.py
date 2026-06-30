import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = Path("anime_data.pkl")
VECTORIZER_PATH = Path("vectorizer.pkl")
TFIDF_MATRIX_PATH = Path("tfidf_matrix.pkl")


def load_artifacts():
    with DATA_PATH.open("rb") as f:
        anime_df = pickle.load(f)

    with VECTORIZER_PATH.open("rb") as f:
        vectorizer = pickle.load(f)

    with TFIDF_MATRIX_PATH.open("rb") as f:
        tfidf_matrix = pickle.load(f)

    return anime_df, vectorizer, tfidf_matrix


anime_df, vectorizer, tfidf_matrix = load_artifacts()


def get_recommendations(title, top_n=5):
    title = title.strip()
    title_matches = anime_df[anime_df["title"].astype(str).str.lower() == title.lower()]

    if title_matches.empty:
        raise ValueError(f"Anime '{title}' was not found in the dataset.")

    anime_index = title_matches.index[0]
    requested_vector = tfidf_matrix[anime_index]
    similarity_scores = cosine_similarity(requested_vector, tfidf_matrix).flatten()

    similarity_scores[anime_index] = -1
    top_indices = similarity_scores.argsort()[::-1][:top_n]

    recommendations = anime_df.iloc[top_indices][["title", "genre", "score", "episodes"]].copy()
    recommendations["similarity_score"] = similarity_scores[top_indices]

    print(f"\nTop {top_n} recommendations for '{title}':")
    print(recommendations.to_string(index=False))
    return recommendations


def plan_binge_watch(recommended_anime_df, max_hours):
    if recommended_anime_df is None or recommended_anime_df.empty:
        return []

    anime_subset = recommended_anime_df.copy()
    anime_subset["estimated_hours"] = anime_subset["episodes"].fillna(0).astype(float) * 0.4
    anime_subset = anime_subset.sort_values(by="score", ascending=False).reset_index(drop=True)

    watchlist = []
    total_hours = 0.0
    total_score = 0.0

    for _, row in anime_subset.iterrows():
        if total_hours + row["estimated_hours"] <= max_hours:
            watchlist.append(row)
            total_hours += row["estimated_hours"]
            total_score += row["score"]

    if watchlist:
        plan_df = pd.DataFrame(watchlist)
        print("\nGreedy Binge-Watch Plan")
        print("=" * 40)
        print(plan_df[["title", "score", "episodes", "estimated_hours"]].to_string(index=False))
        print("=" * 40)
        print(f"Total scheduled hours: {total_hours:.2f}")
        print(f"Average rating score: {plan_df['score'].mean():.2f}")
        return plan_df

    print("\nNo anime could fit within the given time budget.")
    return pd.DataFrame(columns=["title", "score", "episodes", "estimated_hours"])


if __name__ == "__main__":
    try:
        recommendations = get_recommendations("Haikyuu!! Second Season")
        plan_binge_watch(recommendations, 8)
    except ValueError as exc:
        print(f"Lookup failed: {exc}")
