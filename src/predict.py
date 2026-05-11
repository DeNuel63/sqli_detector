"""
SQL Injection Detection — Inference
=====================================
Loads the trained model and TF-IDF vectoriser, rebuilds hand-crafted
features, and classifies one or more raw query strings.

Input  : models/model.pkl
         models/tfidf_vectorizer.pkl
Output : prediction printed to console (and returned as dict)

Usage — from the command line:
    # Single query
    python src/predict.py --query "SELECT * FROM users WHERE id=1 OR 1=1--"

    # Multiple queries from a .txt file (one per line)
    python src/predict.py --file queries.txt

    # Override the default classification threshold
    python src/predict.py --query "..." --threshold 0.35

Usage — as an imported module:
    from src.predict import load_pipeline, predict

    pipeline = load_pipeline()
    result   = predict("SELECT * FROM users WHERE id=1 OR 1=1--", pipeline)
    print(result)

⚠  CLASS IMBALANCE & THRESHOLD REMINDER
    The model was trained on 57% SQLi / 43% Benign data.
    Real production traffic is 99%+ Benign.
    Default threshold here is 0.35 (lower than 0.5) to reduce
    false negatives (missed attacks) in a benign-heavy environment.
    Tune this value based on evaluate.py's threshold analysis.
"""

import re
import html
import math
import json
import pickle
import argparse
import numpy as np
import pandas as pd

from collections import Counter
from scipy.sparse import hstack, csr_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

MODELS_DIR = "models"
MODEL_PATH = f"{MODELS_DIR}/model.pkl"
VEC_PATH   = f"{MODELS_DIR}/tfidf_vectorizer.pkl"

# ⚠  Default threshold — intentionally below 0.5 for production use.
#    Lower = catch more attacks, higher = fewer false alarms.
#    See evaluate.py Step 6b for how to derive the right value for your traffic.
DEFAULT_THRESHOLD = 0.35


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Cleaning (mirrors data_cleaning.py, applied per query at runtime)
# ─────────────────────────────────────────────────────────────────────────────

def clean_query(text: str) -> str:
    """Apply the same cleaning steps used in data_cleaning.py."""

    # Coerce to string
    text = str(text)

    # Decode HTML entities (&amp; → &, &#39; → ', etc.)
    text = html.unescape(text)

    # Decode percent-encoded sequences (%27 → ', %20 → space)
    def _decode_percent(match):
        try:
            return bytes.fromhex(match.group(1)).decode("utf-8", errors="replace")
        except ValueError:
            return match.group(0)
    text = re.sub(r"%([0-9a-fA-F]{2})", _decode_percent, text)

    # Normalise line endings, strip each line, collapse whitespace
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = " ".join(line for line in lines if line)

    # Strip null bytes and control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Hand-crafted features (mirrors features.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

SQL_KEYWORDS = [
    "select", "insert", "update", "delete", "drop", "create", "alter",
    "truncate", "replace", "merge", "from", "where", "having", "group by",
    "order by", "limit", "offset", "union", "join", "inner join",
    "left join", "right join", "or", "and", "not", "null", "is null",
    "is not null", "like", "between", "in", "exists", "case", "when",
    "then", "else", "sleep", "benchmark", "waitfor", "delay", "char",
    "ascii", "hex", "unhex", "concat", "substring", "substr", "mid",
    "length", "count", "version", "database", "schema", "load_file",
    "outfile", "dumpfile", "convert", "cast", "coalesce", "ifnull", "iif",
    "exec", "execute", "sp_", "xp_", "sys.", "information_schema",
    "--", "#", "/*", "*/", "1=1", "1 = 1", "'a'='a", "or 1", "or true",
]

SPECIAL_CHARS = {
    "single_quote":  "'",
    "double_quote":  '"',
    "semicolon":     ";",
    "dash_dash":     "--",
    "hash":          "#",
    "open_paren":    "(",
    "close_paren":   ")",
    "open_comment":  "/*",
    "close_comment": "*/",
    "equals":        "=",
    "backslash":     "\\",
    "pipe":          "|",
    "percent":       "%",
    "asterisk":      "*",
    "at_sign":       "@",
    "caret":         "^",
}

OBFUSCATION_PATTERNS = {
    "mixed_case_select":    r"(?i)s[Ee][Ll][Ee][Cc][Tt]",
    "inline_comment":       r"/\*.*?\*/",
    "char_function":        r"(?i)char\s*\(",
    "hex_value":            r"0x[0-9a-fA-F]+",
    "double_encoded_quote": r"%2[57]|%27|%22",
    "null_byte_encoded":    r"%00",
    "tautology_pattern":    r"(?i)(or|and)\s+[\d'\"]+\s*=\s*[\d'\"]+",
    "stacked_queries":      r";\s*(select|insert|update|delete|drop|exec)",
    "time_based_blind":     r"(?i)(sleep|benchmark|waitfor|pg_sleep)\s*\(",
    "union_select":         r"(?i)union\s+(all\s+)?select",
    "comment_terminator":   r"--\s*$|#\s*$",
}

def _shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    freq = Counter(text)
    n    = len(text)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())

def build_handcrafted_features(query: str) -> dict:
    """Rebuild every hand-crafted feature used during training."""
    feats = {}
    text_lower = query.lower()
    n = max(len(query), 1)

    # SQL keyword counts
    for kw in SQL_KEYWORDS:
        col = f"kw_{kw.replace(' ', '_')}"
        feats[col] = text_lower.count(kw)

    # Special character counts + ratios
    for name, ch in SPECIAL_CHARS.items():
        count = query.count(ch)
        feats[f"sc_count_{name}"] = count
        feats[f"sc_ratio_{name}"] = count / n

    # Structural features
    words = query.split()
    n_word = max(len(words), 1)
    feats["char_len"]          = len(query)
    feats["word_count"]        = n_word
    feats["digit_ratio"]       = sum(c.isdigit() for c in query) / n
    feats["punct_ratio"]       = sum(not c.isalnum() and c != " " for c in query) / n
    feats["upper_ratio"]       = sum(c.isupper() for c in query) / n
    feats["avg_word_len"]      = sum(len(w) for w in words) / n_word
    feats["unique_char_count"] = len(set(query))
    feats["paren_balance"]     = abs(query.count("(") - query.count(")"))
    feats["sq_unmatched"]      = query.count("'") % 2
    feats["dq_unmatched"]      = query.count('"') % 2

    # Entropy features
    feats["entropy_overall"] = _shannon_entropy(query)
    alnum_only = re.sub(r"[^a-zA-Z0-9]", "", query)
    feats["entropy_alnum"] = _shannon_entropy(alnum_only) if alnum_only else 0.0

    # Obfuscation pattern flags
    for name, pattern in OBFUSCATION_PATTERNS.items():
        feats[name] = int(bool(re.search(pattern, query, re.IGNORECASE)))

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Pipeline loader
# ─────────────────────────────────────────────────────────────────────────────

def load_pipeline(model_path: str = MODEL_PATH,
                  vec_path: str   = VEC_PATH) -> dict:
    """
    Load model and vectoriser from models/.
    Returns a dict so callers can inspect or swap components easily.
    """
    model      = pickle.load(open(model_path, "rb"))
    vectorizer = pickle.load(open(vec_path,   "rb"))
    return {"model": model, "vectorizer": vectorizer}


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Feature builder for inference
# Mirrors the combine_features() step in features.py exactly.
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(queries: list, vectorizer) -> csr_matrix:
    """
    Given a list of raw query strings and a fitted TF-IDF vectoriser,
    return a sparse feature matrix ready for model.predict_proba().

    Pipeline:
      raw query → clean → TF-IDF + hand-crafted → combined sparse matrix
    """
    cleaned = [clean_query(q) for q in queries]

    # TF-IDF features (sparse)
    tfidf_matrix = vectorizer.transform(cleaned)

    # Hand-crafted features (dense → sparse)
    hc_rows   = [build_handcrafted_features(q) for q in cleaned]
    hc_df     = pd.DataFrame(hc_rows).fillna(0)
    hc_sparse = csr_matrix(hc_df.values.astype(np.float32))

    return hstack([tfidf_matrix, hc_sparse], format="csr")


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Predict function
# ─────────────────────────────────────────────────────────────────────────────

def predict(query: str,
            pipeline: dict,
            threshold: float = DEFAULT_THRESHOLD) -> dict:
    """
    Classify a single raw query string.

    Args:
        query     : Raw SQL query string to classify.
        pipeline  : Dict returned by load_pipeline().
        threshold : Classification threshold. Default is 0.35.
                    ⚠  Lower = more sensitive (fewer missed attacks,
                       more false alarms). Tune via evaluate.py Step 6b.

    Returns dict:
        {
            "query":      original query,
            "cleaned":    cleaned version,
            "label":      0 (Benign) or 1 (SQLi),
            "prediction": "Benign" or "SQLi",
            "confidence": float — P(SQLi),
            "threshold":  float — threshold used,
            "risk_level": "Low" / "Medium" / "High",
        }
    """
    model      = pipeline["model"]
    vectorizer = pipeline["vectorizer"]

    X      = build_feature_matrix([query], vectorizer)
    prob   = float(model.predict_proba(X)[0, 1])   # P(SQLi)
    label  = int(prob >= threshold)

    # Risk banding — useful for triage dashboards
    if prob < 0.3:
        risk = "Low"
    elif prob < 0.7:
        risk = "Medium"
    else:
        risk = "High"

    return {
        "query":      query,
        "cleaned":    clean_query(query),
        "label":      label,
        "prediction": "SQLi" if label == 1 else "Benign",
        "confidence": round(prob, 4),
        "threshold":  threshold,
        "risk_level": risk,
    }


def predict_batch(queries: list,
                  pipeline: dict,
                  threshold: float = DEFAULT_THRESHOLD) -> pd.DataFrame:
    """
    Classify a list of raw query strings in one pass.
    More efficient than calling predict() in a loop for large batches.

    Returns a DataFrame with one row per query and the same fields
    as predict() plus an index column.
    """
    model      = pipeline["model"]
    vectorizer = pipeline["vectorizer"]

    X     = build_feature_matrix(queries, vectorizer)
    probs = model.predict_proba(X)[:, 1]
    labels = (probs >= threshold).astype(int)

    def _risk(p):
        return "Low" if p < 0.3 else "Medium" if p < 0.7 else "High"

    return pd.DataFrame({
        "query":      queries,
        "cleaned":    [clean_query(q) for q in queries],
        "label":      labels,
        "prediction": ["SQLi" if l == 1 else "Benign" for l in labels],
        "confidence": [round(float(p), 4) for p in probs],
        "threshold":  threshold,
        "risk_level": [_risk(p) for p in probs],
    })


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="SQL Injection Detector — classify one or more queries."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", type=str,
                       help="Single query string to classify.")
    group.add_argument("--file",  type=str,
                       help="Path to a .txt file with one query per line.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Classification threshold (default: {DEFAULT_THRESHOLD}). "
                             "Lower = more sensitive.")
    parser.add_argument("--model",     type=str, default=MODEL_PATH)
    parser.add_argument("--vectorizer",type=str, default=VEC_PATH)
    args = parser.parse_args()

    # Load pipeline
    pipeline = load_pipeline(args.model, args.vectorizer)
    print(f"\nModel loaded     : {args.model}")
    print(f"Vectorizer loaded: {args.vectorizer}")
    print(f"Threshold        : {args.threshold}")
    print(f"{'─' * 60}\n")

    # Single query
    if args.query:
        result = predict(args.query, pipeline, threshold=args.threshold)
        print(f"Query      : {result['query']}")
        print(f"Cleaned    : {result['cleaned']}")
        print(f"Prediction : {result['prediction']}")
        print(f"Confidence : {result['confidence']}  (P(SQLi))")
        print(f"Risk level : {result['risk_level']}")
        print(f"Threshold  : {result['threshold']}")

    # Batch from file
    else:
        with open(args.file) as f:
            queries = [line.strip() for line in f if line.strip()]

        print(f"Classifying {len(queries):,} queries from {args.file}...\n")
        results_df = predict_batch(queries, pipeline, threshold=args.threshold)

        print(results_df.to_string(index=False))
        print(f"\n{'─' * 60}")
        print(f"Total      : {len(results_df):,}")
        print(f"SQLi found : {(results_df['label']==1).sum():,}")
        print(f"Benign     : {(results_df['label']==0).sum():,}")

        # ⚠  Class imbalance reminder at runtime
        sqli_pct = (results_df['label']==1).mean() * 100
        if sqli_pct > 20:
            print(f"\n⚠  {sqli_pct:.1f}% of these queries flagged as SQLi.")
            print(f"   If this is production traffic, that rate is unusually high.")
            print(f"   Consider raising the threshold (currently {args.threshold}).")
        else:
            print(f"\n✓  {sqli_pct:.1f}% SQLi rate looks reasonable for production traffic.")