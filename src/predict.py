"""
SQL Injection Detection - Inference.

Loads the trained model and TF-IDF vectorizer, applies the same runtime
cleaning used during data preparation, then builds features through
src.features so training and inference cannot drift apart.
"""

import argparse
import html
import pickle
import re

import pandas as pd
from scipy.sparse import csr_matrix

from src.features import (
    apply_tfidf,
    build_handcrafted_features,
    combine_features,
    queries_to_handcrafted_matrix,
)


MODELS_DIR = "models"
MODEL_PATH = f"{MODELS_DIR}/model.pkl"
VEC_PATH = f"{MODELS_DIR}/tfidf_vectorizer.pkl"

# Lower thresholds catch more attacks; higher thresholds reduce false alarms.
DEFAULT_THRESHOLD = 0.35


def clean_query(text: str) -> str:
    """Apply the runtime cleaning used before inference."""
    text = str(text)
    text = html.unescape(text)

    def _decode_percent(match):
        try:
            return bytes.fromhex(match.group(1)).decode("utf-8", errors="replace")
        except ValueError:
            return match.group(0)

    text = re.sub(r"%([0-9a-fA-F]{2})", _decode_percent, text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = " ".join(line for line in lines if line)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip()


def load_pipeline(model_path: str = MODEL_PATH, vec_path: str = VEC_PATH) -> dict:
    """Load the trained model and fitted vectorizer from disk."""
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    with open(vec_path, "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return {"model": model, "vectorizer": vectorizer}


def build_feature_matrix(queries: list, vectorizer) -> csr_matrix:
    """
    Build the sparse inference feature matrix.

    The handcrafted feature builder and final feature stacking are imported
    from src.features, which is also used during training.
    """
    cleaned = [clean_query(q) for q in queries]
    tfidf_matrix = apply_tfidf(vectorizer, cleaned)
    handcrafted_df = queries_to_handcrafted_matrix(cleaned)
    return combine_features(tfidf_matrix, handcrafted_df)


def _risk_level(probability: float) -> str:
    if probability < 0.3:
        return "Low"
    if probability < 0.7:
        return "Medium"
    return "High"


def predict(query: str, pipeline: dict, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Classify a single raw query string."""
    model = pipeline["model"]
    vectorizer = pipeline["vectorizer"]

    features = build_feature_matrix([query], vectorizer)
    probability = float(model.predict_proba(features)[0, 1])
    label = int(probability >= threshold)

    return {
        "query": query,
        "cleaned": clean_query(query),
        "label": label,
        "prediction": "SQLi" if label == 1 else "Benign",
        "confidence": round(probability, 4),
        "threshold": threshold,
        "risk_level": _risk_level(probability),
    }


def predict_batch(
    queries: list,
    pipeline: dict,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Classify a list of raw query strings in one model pass."""
    model = pipeline["model"]
    vectorizer = pipeline["vectorizer"]

    features = build_feature_matrix(queries, vectorizer)
    probabilities = model.predict_proba(features)[:, 1]
    labels = (probabilities >= threshold).astype(int)

    return pd.DataFrame(
        {
            "query": queries,
            "cleaned": [clean_query(q) for q in queries],
            "label": labels,
            "prediction": ["SQLi" if label == 1 else "Benign" for label in labels],
            "confidence": [round(float(prob), 4) for prob in probabilities],
            "threshold": threshold,
            "risk_level": [_risk_level(float(prob)) for prob in probabilities],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SQL Injection Detector - classify one or more queries."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", type=str, help="Single query string to classify.")
    group.add_argument("--file", type=str, help="Path to a text file with one query per line.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Classification threshold (default: {DEFAULT_THRESHOLD}). Lower is more sensitive.",
    )
    parser.add_argument("--model", type=str, default=MODEL_PATH)
    parser.add_argument("--vectorizer", type=str, default=VEC_PATH)
    args = parser.parse_args()

    pipeline = load_pipeline(args.model, args.vectorizer)
    print(f"\nModel loaded     : {args.model}")
    print(f"Vectorizer loaded: {args.vectorizer}")
    print(f"Threshold        : {args.threshold}")
    print(f"{'-' * 60}\n")

    if args.query:
        result = predict(args.query, pipeline, threshold=args.threshold)
        print(f"Query      : {result['query']}")
        print(f"Cleaned    : {result['cleaned']}")
        print(f"Prediction : {result['prediction']}")
        print(f"Confidence : {result['confidence']}  (P(SQLi))")
        print(f"Risk level : {result['risk_level']}")
        print(f"Threshold  : {result['threshold']}")
        return

    with open(args.file, encoding="utf-8") as query_file:
        queries = [line.strip() for line in query_file if line.strip()]

    print(f"Classifying {len(queries):,} queries from {args.file}...\n")
    results_df = predict_batch(queries, pipeline, threshold=args.threshold)

    print(results_df.to_string(index=False))
    print(f"\n{'-' * 60}")
    print(f"Total      : {len(results_df):,}")
    print(f"SQLi found : {(results_df['label'] == 1).sum():,}")
    print(f"Benign     : {(results_df['label'] == 0).sum():,}")

    sqli_pct = (results_df["label"] == 1).mean() * 100
    if sqli_pct > 20:
        print(f"\nWARNING: {sqli_pct:.1f}% of these queries flagged as SQLi.")
        print("   If this is production traffic, that rate is unusually high.")
        print(f"   Consider raising the threshold (currently {args.threshold}).")
    else:
        print(f"\nOK: {sqli_pct:.1f}% SQLi rate looks reasonable for production traffic.")


if __name__ == "__main__":
    main()
