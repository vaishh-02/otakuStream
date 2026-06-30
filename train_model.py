import ast
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


DATA_PATH = Path("animes.csv")
DATA_OUTPUT_PATH = Path("anime_data.pkl")
VECTORIZER_OUTPUT_PATH = Path("vectorizer.pkl")
TFIDF_MATRIX_OUTPUT_PATH = Path("tfidf_matrix.pkl")


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


def build_features_dataframe(file_path: str = "animes.csv") -> pd.DataFrame:
    anime_df = pd.read_csv(file_path)

    anime_df["genre"] = anime_df["genre"].fillna("")
    anime_df["synopsis"] = anime_df["synopsis"].fillna("")

    anime_df["genre"] = anime_df["genre"].apply(clean_genre)

    anime_df["features"] = (
        anime_df["genre"].astype(str) + " " + anime_df["synopsis"].astype(str)
    ).str.strip()

    return anime_df


def train_and_save_models():
    anime_df = build_features_dataframe(str(DATA_PATH))

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(anime_df["features"])

    with DATA_OUTPUT_PATH.open("wb") as f:
        pickle.dump(anime_df, f)

    with VECTORIZER_OUTPUT_PATH.open("wb") as f:
        pickle.dump(vectorizer, f)

    with TFIDF_MATRIX_OUTPUT_PATH.open("wb") as f:
        pickle.dump(tfidf_matrix, f)

    print("Saved anime data and model artifacts successfully:")
    print(f"- {DATA_OUTPUT_PATH}")
    print(f"- {VECTORIZER_OUTPUT_PATH}")
    print(f"- {TFIDF_MATRIX_OUTPUT_PATH}")


if __name__ == "__main__":
    train_and_save_models()
