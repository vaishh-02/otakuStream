import ast
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

file_path = "animes.csv"
anime_df = pd.read_csv(file_path)

anime_df["genre"] = anime_df["genre"].fillna("")
anime_df["synopsis"] = anime_df["synopsis"].fillna("")


def clean_genre(value):
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, (list, tuple)):
                    return ", ".join(str(item).strip() for item in parsed)
            except (ValueError, SyntaxError):
                pass
    return value

anime_df["genre"] = anime_df["genre"].apply(clean_genre)

anime_df["features"] = (
    anime_df["genre"].fillna("").astype(str) + " " + anime_df["synopsis"].fillna("").astype(str)
).str.strip()

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(anime_df["features"])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


def get_recommendations(title, top_n=5):
    title = title.strip()
    matches = anime_df[anime_df["title"].astype(str).str.lower() == title.lower()]

    if matches.empty:
        raise ValueError(f"Anime '{title}' not found in the dataset.")

    anime_index = matches.index[0]
    sim_scores = list(enumerate(cosine_sim[anime_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [score for score in sim_scores if score[0] != anime_index]

    seen_titles = set()
    unique_scores = []
    for idx, score in sim_scores:
        candidate_title = anime_df.at[idx, "title"]
        if candidate_title not in seen_titles:
            seen_titles.add(candidate_title)
            unique_scores.append((idx, score))
        if len(unique_scores) == top_n:
            break

    recommended_anime = anime_df.iloc[[idx for idx, _ in unique_scores]][["title", "genre", "score"]].copy()
    recommended_anime["similarity_score"] = [score for _, score in unique_scores]

    print(f"\nTop {top_n} recommendations for '{title}':")
    print(recommended_anime.to_string(index=False))
    return recommended_anime


print("Data shape (rows, columns):", anime_df.shape)
print("\nFirst 3 rows:")
print(anime_df.head(3).to_string(index=False))
