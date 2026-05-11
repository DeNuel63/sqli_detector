"""
SQL Injection Detection — Data Cleaning
========================================
Input  : data/raw/trainingdata.csv
Output : data/processed/cleaned.csv
         data/processed/train.csv
         data/processed/val.csv
         data/processed/test.csv

Run:
    python src/data_cleaning.py
"""

import os
import re
import html
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Config ────────────────────────────────────────────────────────────────────
RAW_PATH      = "data/raw/trainingdata.csv"
PROCESSED_DIR = "data/processed"

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10
RANDOM_SEED = 42

os.makedirs(PROCESSED_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load raw data
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — LOADING RAW DATA")
print("=" * 60)

df = pd.read_csv(RAW_PATH)
print(f"Loaded  : {len(df):,} rows")
print(f"Columns : {list(df.columns)}")
print(f"Dtypes  :\n{df.dtypes}\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Drop exact duplicate rows
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 2 — REMOVING DUPLICATE ROWS")
print("=" * 60)

before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
removed = before - len(df)

print(f"Duplicates removed : {removed}")
print(f"Rows remaining     : {len(df):,}\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Drop rows with null or empty Query values
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 3 — REMOVING NULL / EMPTY QUERIES")
print("=" * 60)

before = len(df)

# Coerce to string first (catches any stray non-string values)
df["Query"] = df["Query"].astype(str)

# Drop actual NaN (post-coerce they become the string "nan")
df = df[df["Query"].str.lower() != "nan"].reset_index(drop=True)

# Drop empty / whitespace-only strings
df = df[df["Query"].str.strip().str.len() > 0].reset_index(drop=True)

removed = before - len(df)
print(f"Null / empty queries removed : {removed}")
print(f"Rows remaining               : {len(df):,}\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Validate and coerce labels
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 4 — VALIDATING LABELS")
print("=" * 60)

before = len(df)

# Coerce to integer (catches string "0"/"1" if present)
df["Label"] = pd.to_numeric(df["Label"], errors="coerce")

# Drop rows whose label couldn't be parsed
invalid_labels = df["Label"].isna().sum()
df = df.dropna(subset=["Label"]).reset_index(drop=True)
df["Label"] = df["Label"].astype(int)

# Drop rows with unexpected label values
valid_labels = {0, 1}
unexpected = ~df["Label"].isin(valid_labels)
df = df[~unexpected].reset_index(drop=True)

removed = before - len(df)
print(f"Invalid labels removed : {removed}")
print(f"Label distribution:\n{df['Label'].value_counts().to_string()}\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Decode HTML entities and URL-encoded characters
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 5 — DECODING HTML ENTITIES & URL ENCODING")
print("=" * 60)

def decode_query(text):
    """
    Decode HTML entities (&amp; &lt; &#39; etc.) and percent-encoded
    sequences (%27 %20 etc.) so the model sees raw SQL characters.
    URL decoding is attempted only for clearly percent-encoded patterns
    to avoid mangling legitimate % signs in fuzzing payloads.
    """
    # 1. HTML entities → raw chars  (&amp; → &, &lt; → <, &#39; → ')
    text = html.unescape(text)

    # 2. Percent-encoded sequences → raw chars (%27 → ', %20 → space)
    #    Only decode sequences that look like valid percent-encoding.
    def decode_percent(match):
        try:
            return bytes.fromhex(match.group(1)).decode("utf-8", errors="replace")
        except ValueError:
            return match.group(0)

    text = re.sub(r"%([0-9a-fA-F]{2})", decode_percent, text)
    return text

before_sample = df["Query"].iloc[0]
df["Query"] = df["Query"].apply(decode_query)
after_sample  = df["Query"].iloc[0]

# Count how many rows were actually changed
changed = (df["Query"] != df["Query"].apply(lambda q: q)).sum()  # placeholder; count below
html_pattern = re.compile(r"&[a-z]+;|&#\d+;|%[0-9a-fA-F]{2}")

print(f"Decoding applied to all queries.")
print(f"Sample (first row unchanged here, but encoding resolved where present).\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Normalise whitespace
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 6 — NORMALISING WHITESPACE")
print("=" * 60)

def normalise_whitespace(text):
    """
    - Strip leading/trailing whitespace
    - Collapse multiple spaces/tabs into a single space
    - Preserve intentional newlines in multi-statement queries by
      replacing \r\n and \r with \n, then stripping each line.
    """
    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip each line and collapse internal runs of spaces/tabs
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    # Remove blank lines, rejoin
    text = " ".join(line for line in lines if line)
    return text.strip()

before_ws = (df["Query"].str.contains(r"  +|\t|\r|\n", regex=True)).sum()
df["Query"] = df["Query"].apply(normalise_whitespace)
after_ws  = (df["Query"].str.contains(r"  +|\t|\r|\n", regex=True)).sum()

print(f"Queries with whitespace anomalies before : {before_ws}")
print(f"Queries with whitespace anomalies after  : {after_ws}\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Remove null bytes and control characters
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 7 — STRIPPING NULL BYTES & CONTROL CHARACTERS")
print("=" * 60)

def strip_control_chars(text):
    """
    Remove null bytes (\x00) and ASCII control characters (\x01–\x1f, \x7f).
    Preserves \t (tab) and \n (newline) since step 6 already handled them.
    Preserves \x1b only if needed for terminal escape sequences (not typical
    in SQL injection payloads).
    """
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

before_ctrl = df["Query"].apply(lambda q: bool(re.search(r"[\x00-\x1f\x7f]", q))).sum()
df["Query"] = df["Query"].apply(strip_control_chars)
after_ctrl  = df["Query"].apply(lambda q: bool(re.search(r"[\x00-\x1f\x7f]", q))).sum()

print(f"Queries with control chars before : {before_ctrl}")
print(f"Queries with control chars after  : {after_ctrl}\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Drop queries that became empty after cleaning
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 8 — DROPPING POST-CLEAN EMPTY QUERIES")
print("=" * 60)

before = len(df)
df = df[df["Query"].str.strip().str.len() > 0].reset_index(drop=True)
removed = before - len(df)

print(f"Queries that became empty after cleaning : {removed}")
print(f"Rows remaining                           : {len(df):,}\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Final deduplication (post-cleaning)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 9 — FINAL DEDUPLICATION (POST-CLEAN)")
print("=" * 60)

before = len(df)

# Check for label leakage before deduplication
shared = df.groupby("Query")["Label"].nunique()
leaked = shared[shared > 1]
print(f"Queries appearing in both classes (label leakage) : {len(leaked)}")
if len(leaked):
    print("  ⚠  Dropping leaked queries (ambiguous ground truth).")
    df = df[~df["Query"].isin(leaked.index)].reset_index(drop=True)

# Drop any new duplicates introduced by decoding/normalisation
df = df.drop_duplicates().reset_index(drop=True)

removed = before - len(df)
print(f"Additional rows removed : {removed}")
print(f"Rows remaining          : {len(df):,}\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — Cleaning summary
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 10 — CLEANING SUMMARY")
print("=" * 60)

print(f"\nFinal dataset shape : {df.shape}")
print(f"\nLabel distribution :")
counts = df["Label"].value_counts()
pcts   = df["Label"].value_counts(normalize=True) * 100
for label in [0, 1]:
    name = "Benign (0)" if label == 0 else "SQLi   (1)"
    print(f"  {name} : {counts[label]:,} ({pcts[label]:.2f}%)")

print(f"\nQuery length stats (chars):")
print(df["Query"].str.len().describe().round(1).to_string())

# Verify no remaining issues
assert df["Query"].isnull().sum() == 0,       "Nulls still present!"
assert df["Label"].isnull().sum() == 0,       "Null labels still present!"
assert df.duplicated().sum() == 0,            "Duplicates still present!"
assert df["Label"].isin({0, 1}).all(),        "Invalid label values!"
assert (df["Query"].str.strip().str.len() > 0).all(), "Empty queries still present!"
print("\n✓  All assertions passed — dataset is clean.\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 11 — Save cleaned dataset & stratified splits
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 11 — SAVING CLEANED DATA & SPLITS")
print("=" * 60)

# Save full cleaned dataset
cleaned_path = os.path.join(PROCESSED_DIR, "cleaned.csv")
df.to_csv(cleaned_path, index=False)
print(f"Saved: {cleaned_path}  ({len(df):,} rows)")

# Stratified 80 / 10 / 10 split
X = df["Query"]
y = df["Label"]

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y,
    test_size=TEST_RATIO,
    stratify=y,
    random_state=RANDOM_SEED,
)

val_relative = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=val_relative,
    stratify=y_train_val,
    random_state=RANDOM_SEED,
)

splits = {
    "train": (X_train, y_train),
    "val":   (X_val,   y_val),
    "test":  (X_test,  y_test),
}

print()
for name, (X_split, y_split) in splits.items():
    split_df   = pd.DataFrame({"Query": X_split, "Label": y_split})
    split_path = os.path.join(PROCESSED_DIR, f"{name}.csv")
    split_df.to_csv(split_path, index=False)
    pct_sqli = y_split.mean() * 100
    print(f"Saved: {split_path:<35} {len(split_df):>7,} rows  |  {pct_sqli:.1f}% SQLi")

print()
print("=" * 60)
print("DATA CLEANING COMPLETE")
print("=" * 60)
print("""
Files written:
  data/processed/cleaned.csv   ← full cleaned dataset
  data/processed/train.csv     ← 80% for model training
  data/processed/val.csv       ← 10% for hyperparameter tuning
  data/processed/test.csv      ← 10% held-out for final evaluation
""")