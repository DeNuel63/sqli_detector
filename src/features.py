"""
SQL Injection Detection — Feature Engineering
===============================================
Input  : data/processed/cleaned.csv  (or train/val/test splits)
Output : data/processed/features_train.pkl
         data/processed/features_val.pkl
         data/processed/features_test.pkl
         data/processed/tfidf_vectorizer.pkl
         data/processed/feature_names.json

Run:
    python src/features.py

⚠  CLASS IMBALANCE REMINDER
    Your dataset is 57% SQLi / 43% benign (1.3:1 ratio).
    This is mild but keep it in mind:
    - During model training (Step 5) use class_weight='balanced'
    - After training (Step 6) check recall for class 0 (benign)
    - If benign recall is below ~0.85, revisit with SMOTE
    This file does NOT apply any resampling — that belongs in train.py.
"""

import os
import re
import json
import math
import pickle

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, save_npz, csr_matrix

# ── Config ────────────────────────────────────────────────────────────────────
PROCESSED_DIR   = "data/processed"
TFIDF_MAX_FEATS = 10_000   # covers ~95% of vocabulary
TFIDF_NGRAM     = (1, 3)   # unigrams, bigrams, trigrams
RANDOM_SEED     = 42

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP 1 — SQL keyword counts
# Most discriminative single signal: SQLi payloads are dense with SQL keywords
# ─────────────────────────────────────────────────────────────────────────────

SQL_KEYWORDS = [
    # DML / DDL
    "select", "insert", "update", "delete", "drop", "create", "alter",
    "truncate", "replace", "merge",
    # Clauses
    "from", "where", "having", "group by", "order by", "limit", "offset",
    "union", "join", "inner join", "left join", "right join",
    # Logic / comparison
    "or", "and", "not", "null", "is null", "is not null",
    "like", "between", "in", "exists", "case", "when", "then", "else",
    # Functions commonly abused
    "sleep", "benchmark", "waitfor", "delay",
    "char", "ascii", "hex", "unhex", "concat", "substring", "substr",
    "mid", "length", "count", "version", "database", "schema",
    "load_file", "outfile", "dumpfile",
    "convert", "cast", "coalesce", "ifnull", "iif",
    # Stored proc / exec
    "exec", "execute", "sp_", "xp_", "sys.", "information_schema",
    # Comment markers
    "--", "#", "/*", "*/",
    # Tautologies
    "1=1", "1 = 1", "'a'='a", "or 1", "or true",
]

def count_sql_keywords(text):
    """Return dict of keyword → count for one query."""
    text_lower = text.lower()
    return {kw: text_lower.count(kw) for kw in SQL_KEYWORDS}


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP 2 — Special character ratios
# SQLi payloads are dense with quotes, dashes, semicolons, etc.
# Ratios (per-character) normalise for query length.
# ─────────────────────────────────────────────────────────────────────────────

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

def special_char_features(text):
    n = max(len(text), 1)
    feats = {}
    for name, ch in SPECIAL_CHARS.items():
        count = text.count(ch)
        feats[f"sc_count_{name}"]  = count
        feats[f"sc_ratio_{name}"]  = count / n
    return feats


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP 3 — Structural / length features
# SQLi payloads tend to be shorter and structurally different from natural text
# ─────────────────────────────────────────────────────────────────────────────

def structural_features(text):
    words  = text.split()
    n_char = len(text)
    n_word = max(len(words), 1)

    # Digit density
    digit_count = sum(c.isdigit() for c in text)

    # Punctuation density (anything not alphanumeric or space)
    punct_count = sum(not c.isalnum() and c != " " for c in text)

    # Uppercase ratio (obfuscated SQLi sometimes uses mixed case)
    upper_count = sum(c.isupper() for c in text)

    # Average word length (SQL keywords are short; NL sentences have longer words)
    avg_word_len = sum(len(w) for w in words) / n_word

    # Unique character count / diversity
    unique_chars = len(set(text))

    # Parenthesis balance (imbalanced parens → likely injection)
    paren_balance = abs(text.count("(") - text.count(")"))

    # Quote balance (unmatched quotes → classic injection signal)
    sq_balance = text.count("'") % 2   # 1 if odd (unmatched), 0 if even
    dq_balance = text.count('"') % 2

    return {
        "char_len":         n_char,
        "word_count":       n_word,
        "digit_ratio":      digit_count / max(n_char, 1),
        "punct_ratio":      punct_count / max(n_char, 1),
        "upper_ratio":      upper_count / max(n_char, 1),
        "avg_word_len":     avg_word_len,
        "unique_char_count": unique_chars,
        "paren_balance":    paren_balance,
        "sq_unmatched":     sq_balance,
        "dq_unmatched":     dq_balance,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP 4 — Entropy
# Random-looking fuzzing payloads have high entropy.
# Natural language sentences have moderate entropy.
# Structured SQL has low-to-moderate entropy.
# ─────────────────────────────────────────────────────────────────────────────

def shannon_entropy(text):
    """Shannon entropy (bits per character)."""
    if not text:
        return 0.0
    freq  = Counter(text)
    n     = len(text)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())

def entropy_features(text):
    overall = shannon_entropy(text)
    # Entropy of only alphanumeric characters (strips noise from operators)
    alnum_only = re.sub(r"[^a-zA-Z0-9]", "", text)
    alnum_ent  = shannon_entropy(alnum_only) if alnum_only else 0.0
    return {
        "entropy_overall": overall,
        "entropy_alnum":   alnum_ent,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP 5 — Obfuscation / evasion patterns
# Attackers try to bypass simple keyword filters with tricks like:
#   SeLeCt, char(83), 0x53454c454354, inline comments, double encoding
# ─────────────────────────────────────────────────────────────────────────────

OBFUSCATION_PATTERNS = {
    "mixed_case_select":   r"(?i)s[Ee][Ll][Ee][Cc][Tt]",
    "inline_comment":      r"/\*.*?\*/",
    "char_function":       r"(?i)char\s*\(",
    "hex_value":           r"0x[0-9a-fA-F]+",
    "double_encoded_quote":r"%2[57]|%27|%22",   # URL-encoded quotes
    "null_byte_encoded":   r"%00",
    "tautology_pattern":   r"(?i)(or|and)\s+[\d'\"]+\s*=\s*[\d'\"]+",
    "stacked_queries":     r";\s*(select|insert|update|delete|drop|exec)",
    "time_based_blind":    r"(?i)(sleep|benchmark|waitfor|pg_sleep)\s*\(",
    "union_select":        r"(?i)union\s+(all\s+)?select",
    "comment_terminator":  r"--\s*$|#\s*$",
}

def obfuscation_features(text):
    return {
        name: int(bool(re.search(pattern, text, re.IGNORECASE)))
        for name, pattern in OBFUSCATION_PATTERNS.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# MASTER FEATURE BUILDER
# Combines all 5 hand-crafted feature groups into one flat dict per query.
# ─────────────────────────────────────────────────────────────────────────────

def build_handcrafted_features(query):
    feats = {}
    feats.update({f"kw_{k.replace(' ', '_')}": v
                  for k, v in count_sql_keywords(query).items()})
    feats.update(special_char_features(query))
    feats.update(structural_features(query))
    feats.update(entropy_features(query))
    feats.update(obfuscation_features(query))
    return feats

def queries_to_handcrafted_matrix(queries):
    """Convert a list/Series of query strings → (n_samples, n_features) DataFrame."""
    rows = [build_handcrafted_features(q) for q in queries]
    return pd.DataFrame(rows).fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF VECTORISER (character n-grams + word n-grams)
# Fits ONLY on training data to prevent data leakage into val/test.
#
# ⚠  LEAKAGE REMINDER
#    The TF-IDF vectoriser must be fit on train data only, then
#    applied (transform only) to val and test. Fitting on the full
#    dataset would leak test-set vocabulary into the model — giving
#    you inflated evaluation scores that won't hold in production.
# ─────────────────────────────────────────────────────────────────────────────

def fit_tfidf(train_queries):
    """Fit TF-IDF on training queries. Returns fitted vectoriser."""
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=TFIDF_NGRAM,
        max_features=TFIDF_MAX_FEATS,
        sublinear_tf=True,       # log(1 + tf) — dampens very high counts
        min_df=2,                # ignore terms appearing in only 1 document
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w+\b|--|#|/\*|\*/|=|;|'",
                                 # extended pattern captures SQL operators
    )
    vectorizer.fit(train_queries)
    return vectorizer

def apply_tfidf(vectorizer, queries):
    """Transform queries using a fitted vectoriser."""
    return vectorizer.transform(queries)


# ─────────────────────────────────────────────────────────────────────────────
# COMBINE: TF-IDF sparse matrix + hand-crafted dense features
# ─────────────────────────────────────────────────────────────────────────────

def combine_features(tfidf_matrix, handcrafted_df):
    """Horizontally stack sparse TF-IDF with dense hand-crafted features."""
    dense_sparse = csr_matrix(handcrafted_df.values.astype(np.float32))
    return hstack([tfidf_matrix, dense_sparse], format="csr")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — build and save all feature matrices
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    # ── Load splits ───────────────────────────────────────────────────────────
    print("\nLoading cleaned splits...")
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    val_df   = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))
    test_df  = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"  {name:<6}: {len(df):,} rows")

    # ── Hand-crafted features ─────────────────────────────────────────────────
    print("\nBuilding hand-crafted features...")

    hc_train = queries_to_handcrafted_matrix(train_df["Query"])
    hc_val   = queries_to_handcrafted_matrix(val_df["Query"])
    hc_test  = queries_to_handcrafted_matrix(test_df["Query"])

    print(f"  Hand-crafted feature count : {hc_train.shape[1]}")
    print(f"  Sample features            : {list(hc_train.columns[:8])}")

    # ── TF-IDF features ───────────────────────────────────────────────────────
    print("\nFitting TF-IDF vectoriser on TRAIN only...")
    print("  ⚠  Fitting on train only — no leakage into val/test")

    vectorizer    = fit_tfidf(train_df["Query"])
    tfidf_train   = apply_tfidf(vectorizer, train_df["Query"])
    tfidf_val     = apply_tfidf(vectorizer, val_df["Query"])
    tfidf_test    = apply_tfidf(vectorizer, test_df["Query"])

    print(f"  TF-IDF vocabulary size     : {len(vectorizer.vocabulary_):,}")
    print(f"  TF-IDF feature count       : {tfidf_train.shape[1]:,}")

    # ── Combine features ──────────────────────────────────────────────────────
    print("\nCombining TF-IDF + hand-crafted features...")

    X_train = combine_features(tfidf_train, hc_train)
    X_val   = combine_features(tfidf_val,   hc_val)
    X_test  = combine_features(tfidf_test,  hc_test)

    y_train = train_df["Label"].values
    y_val   = val_df["Label"].values
    y_test  = test_df["Label"].values

    total_feats = X_train.shape[1]
    print(f"  Total feature count        : {total_feats:,}")
    print(f"  X_train shape              : {X_train.shape}")
    print(f"  X_val shape                : {X_val.shape}")
    print(f"  X_test shape               : {X_test.shape}")

    # ── Feature sanity checks ─────────────────────────────────────────────────
    print("\nRunning feature sanity checks...")

    assert X_train.shape[0] == len(y_train), "Train row count mismatch!"
    assert X_val.shape[0]   == len(y_val),   "Val row count mismatch!"
    assert X_test.shape[0]  == len(y_test),  "Test row count mismatch!"
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1], \
        "Feature count mismatch across splits!"
    assert not np.isnan(X_train.data).any(), "NaN values in X_train!"
    assert not np.isinf(X_train.data).any(), "Inf values in X_train!"

    print("  ✓ All sanity checks passed")

    # ── Save artefacts ────────────────────────────────────────────────────────
    print("\nSaving feature artefacts...")

    # Feature matrices (sparse)
    for name, X, y in [("train", X_train, y_train),
                        ("val",   X_val,   y_val),
                        ("test",  X_test,  y_test)]:
        X_path = os.path.join(PROCESSED_DIR, f"features_{name}.npz")
        y_path = os.path.join(PROCESSED_DIR, f"labels_{name}.npy")
        save_npz(X_path, X)
        np.save(y_path, y)
        print(f"  Saved: {X_path}")
        print(f"  Saved: {y_path}")

    # Fitted TF-IDF vectoriser (needed at inference time)
    vec_path = os.path.join("models", "tfidf_vectorizer.pkl") 
    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"  Saved: {vec_path}")

    # Feature names (for SHAP / interpretability later)
    tfidf_names = [f"tfidf_{w}" for w in vectorizer.get_feature_names_out()]
    hc_names    = list(hc_train.columns)
    all_names   = tfidf_names + hc_names
    names_path  = os.path.join(PROCESSED_DIR, "feature_names.json")
    with open(names_path, "w") as f:
        json.dump(all_names, f)
    print(f"  Saved: {names_path}  ({len(all_names):,} feature names)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 60)
    print(f"""
Feature breakdown:
  TF-IDF (word 1–3 grams)  : {tfidf_train.shape[1]:>6,}
  SQL keyword counts        : {len([k for k in hc_names if k.startswith('kw_')]):>6,}
  Special char counts/ratios: {len([k for k in hc_names if k.startswith('sc_')]):>6,}
  Structural / length       : {len([k for k in hc_names if not k.startswith(('kw_','sc_','ent','mix','inl','cha','hex','dou','nul','tau','sta','tim','uni','com'))])  :>6,}
  Entropy                   : {len([k for k in hc_names if k.startswith('ent')]):>6,}
  Obfuscation patterns      : {len([k for k in hc_names if k in [n for n in OBFUSCATION_PATTERNS]]):>6,}
  ─────────────────────────────────
  TOTAL                     : {total_feats:>6,}

Files written to data/processed/:
  features_train.npz / labels_train.npy
  features_val.npz   / labels_val.npy
  features_test.npz  / labels_test.npy
  tfidf_vectorizer.pkl
  feature_names.json

""")