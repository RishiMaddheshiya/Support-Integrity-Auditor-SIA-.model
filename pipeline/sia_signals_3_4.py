"""
SIA Stage 1c - Signals 3 & 4 + Ablation Table
===============================================
Signal 3: Resolution-Time Regression (severity proxy)
Signal 4: Rule-Based NLP Keyword Density
Ablation: Agreement of each signal with fused mismatch label

Output: pseudo_labeled_tickets.csv (updated), ablation_table.csv
"""

import re
import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack, issparse

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("sia_signals_3_4.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("SIA-Signals")
SEP = "=" * 60

INPUT_CSV  = "pseudo_labeled_tickets.csv"
OUTPUT_CSV = "pseudo_labeled_tickets.csv"   # overwrite with new columns
ABLATION_CSV = "ablation_table.csv"

LABEL_MAP = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
NUM_MAP   = {v: k for k, v in LABEL_MAP.items()}

# Escalation lexicon for Signal 4
ESCALATION_CATALOGUE = [
    ("data loss",          3.0), ("outage",          3.0),
    ("emergency",          3.0), ("cannot access",   2.8),
    ("failure",            2.5), ("not working",     2.5),
    ("critical",           2.5), ("sla",             2.3),
    ("escalate",           2.3), ("broken",          2.0),
    ("down",               2.0), ("urgent",          2.0),
    ("immediately",        1.8), ("error",           1.2),
    ("problem",            1.0), ("issue",           0.8),
]
NEGATION_WORDS = ["not", "never", "no", "unable", "can't", "won't", "doesn't",
                  "didn't", "don't", "isn't", "wasn't"]


def find_col(df, *keywords):
    for col in df.columns:
        if all(kw.lower() in col.lower() for kw in keywords):
            return col
    return None


# ── SIGNAL 3: Resolution-Time Regression ─────────────────────────────
def add_regression_severity(df, col_rt, col_desc, col_channel, col_type):
    log.info("SIGNAL 3 - Resolution-Time Regression")

    if not col_rt:
        log.warning("  No resolution time column found — using fallback (Medium for all)")
        df["regression_severity"] = "Medium"
        return df

    # Parse resolution time to numeric
    def parse_rt(val):
        try:
            return float(str(val).split()[0])
        except Exception:
            return np.nan

    rt_vals = pd.to_numeric(df[col_rt], errors="coerce")
    if rt_vals.isna().all():
        rt_vals = df[col_rt].apply(parse_rt)
    rt_vals = rt_vals.fillna(rt_vals.median() if not rt_vals.isna().all() else 30.0)
    df["_rt_num"] = rt_vals

    # TF-IDF on description
    tfidf = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
    desc_feats = tfidf.fit_transform(df[col_desc].fillna("").astype(str))

    # Encode channel and type
    extra_frames = []
    for col in [col_channel, col_type]:
        if col:
            enc = LabelEncoder()
            encoded = enc.fit_transform(df[col].fillna("Unknown").astype(str))
            extra_frames.append(pd.DataFrame({col: encoded}))

    if extra_frames:
        extra_arr = pd.concat(extra_frames, axis=1).values
        from scipy.sparse import csr_matrix
        X = hstack([desc_feats, csr_matrix(extra_arr)])
    else:
        X = desc_feats

    y = rt_vals.values

    # Train regressor
    log.info(f"  Training GradientBoostingRegressor on {len(df):,} samples...")
    reg = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)

    # Convert sparse to dense for GBR (it doesn't support sparse)
    if issparse(X):
        X_dense = X.toarray()
    else:
        X_dense = np.array(X)

    reg.fit(X_dense, y)
    pred_rt = reg.predict(X_dense)
    df["_pred_rt"] = pred_rt

    # Bucket by percentiles
    p25 = np.percentile(pred_rt, 25)
    p50 = np.percentile(pred_rt, 50)
    p75 = np.percentile(pred_rt, 75)
    log.info(f"  Predicted RT percentiles: P25={p25:.1f}  P50={p50:.1f}  P75={p75:.1f}")

    def rt_to_severity(v):
        if v >= p75: return "Critical"
        if v >= p50: return "High"
        if v >= p25: return "Medium"
        return "Low"

    df["regression_severity"] = df["_pred_rt"].apply(rt_to_severity)

    dist = df["regression_severity"].value_counts()
    log.info("  Regression severity distribution:")
    for lbl in ["Critical", "High", "Medium", "Low"]:
        cnt = dist.get(lbl, 0)
        log.info(f"    {lbl:<10}: {cnt:>6,}  ({cnt/len(df)*100:.1f}%)")

    df = df.drop(columns=["_rt_num", "_pred_rt"], errors="ignore")
    return df


# ── SIGNAL 4: Rule-Based NLP Keyword Density ─────────────────────────
def add_rule_nlp_severity(df, col_subj, col_desc):
    log.info("SIGNAL 4 - Rule-Based NLP Keyword Density")

    def compute_density(subject, description):
        text  = f"{subject} {description}".strip().lower()
        words = text.split()
        total = max(len(words), 1)

        esc_hits = sum(
            len(re.findall(r"\b" + re.escape(p) + r"\b", text)) * w
            for p, w in ESCALATION_CATALOGUE
        )
        neg_hits = sum(
            len(re.findall(r"\b" + re.escape(w) + r"\b", text))
            for w in NEGATION_WORDS
        )
        density = (esc_hits + neg_hits) / total
        return round(density, 4)

    def density_to_label(d):
        if d > 0.08: return "Critical"
        if d > 0.05: return "High"
        if d > 0.02: return "Medium"
        return "Low"

    subjects = df[col_subj].fillna("").astype(str) if col_subj else pd.Series([""] * len(df))
    descs    = df[col_desc].fillna("").astype(str)

    densities = [
        compute_density(subjects.iloc[i], descs.iloc[i])
        for i in range(len(df))
    ]
    df["keyword_density"]  = densities
    df["rule_nlp_severity"] = [density_to_label(d) for d in densities]

    dist = df["rule_nlp_severity"].value_counts()
    log.info("  Rule-NLP severity distribution:")
    for lbl in ["Critical", "High", "Medium", "Low"]:
        cnt = dist.get(lbl, 0)
        log.info(f"    {lbl:<10}: {cnt:>6,}  ({cnt/len(df)*100:.1f}%)")

    return df


# ── ABLATION TABLE ────────────────────────────────────────────────────
def compute_ablation(df):
    log.info(SEP)
    log.info("ABLATION TABLE - Signal agreement with final mismatch label")

    signals = {
        "llm_severity_score"  : "llm_severity_score",
        "cluster_severity"    : "cluster_severity",
        "regression_severity" : "regression_severity",
        "rule_nlp_severity"   : "rule_nlp_severity",
    }

    target_num = df["mismatch"].values
    rows = []

    for signal_name, col in signals.items():
        if col not in df.columns:
            log.warning(f"  Column '{col}' not found — skipping")
            continue

        assigned_num = df.get("assigned_num",
            df[find_col(df, "priority") or find_col(df, "level")].apply(
                lambda x: LABEL_MAP.get(str(x).strip().title(), 2)
            ) if find_col(df, "priority") else pd.Series([2]*len(df))
        )

        inferred_num = df[col].map(LABEL_MAP).fillna(2)
        signal_mismatch = ((inferred_num - assigned_num).abs() >= 2).astype(int).values

        # Agreement = both signal and fused label agree on mismatch/not
        agreement = (signal_mismatch == target_num).mean() * 100

        rows.append({
            "Signal"              : signal_name,
            "Agreement_with_fused": f"{agreement:.1f}%",
        })
        log.info(f"  {signal_name:<25}: {agreement:.1f}%")

    ablation_df = pd.DataFrame(rows)
    ablation_df.to_csv(ABLATION_CSV, index=False)
    log.info(f"  Saved ablation table -> '{ABLATION_CSV}'")
    return ablation_df


# ── MAIN ──────────────────────────────────────────────────────────────
def main():
    log.info(SEP)
    log.info("SIA Stage 1c - Signals 3 & 4 + Ablation Table")
    log.info(SEP)

    if not Path(INPUT_CSV).exists():
        log.error(f"'{INPUT_CSV}' not found. Run sia_clustering.py first.")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    log.info(f"Loaded {len(df):,} rows from '{INPUT_CSV}'")
    log.info(f"Columns: {list(df.columns)}")

    # Identify columns
    col_subj    = find_col(df, "subject")
    col_desc    = find_col(df, "description")
    col_channel = find_col(df, "channel")
    col_type    = find_col(df, "type")
    col_rt      = (find_col(df, "resolution", "minute")
                   or find_col(df, "resolution", "hour")
                   or find_col(df, "resolution", "time"))
    col_prio    = find_col(df, "priority") or find_col(df, "level")

    log.info(f"  Subject  : {col_subj}")
    log.info(f"  Desc     : {col_desc}")
    log.info(f"  Channel  : {col_channel}")
    log.info(f"  Type     : {col_type}")
    log.info(f"  Res Time : {col_rt}")
    log.info(f"  Priority : {col_prio}")

    if not col_desc:
        log.error("Could not find Description column.")
        sys.exit(1)

    # Ensure assigned_num exists
    if "assigned_num" not in df.columns and col_prio:
        df["assigned_num"] = df[col_prio].apply(
            lambda x: LABEL_MAP.get(str(x).strip().title(), 2)
        )

    # Ensure mismatch column exists
    if "mismatch" not in df.columns:
        if "fused_num" in df.columns and "assigned_num" in df.columns:
            df["mismatch"] = ((df["fused_num"] - df["assigned_num"]).abs() >= 2).astype(int)
        else:
            log.error("No 'mismatch' column found. Run sia_clustering.py first.")
            sys.exit(1)

    log.info(SEP)
    df = add_regression_severity(df, col_rt, col_desc, col_channel, col_type)

    log.info(SEP)
    df = add_rule_nlp_severity(df, col_subj, col_desc)

    ablation_df = compute_ablation(df)

    # Save updated CSV
    df.to_csv(OUTPUT_CSV, index=False)
    log.info(SEP)
    log.info(f"Saved {len(df):,} rows -> '{OUTPUT_CSV}'")
    log.info("Stage 1c COMPLETE")

    log.info("\nAblation Table:")
    log.info(ablation_df.to_string(index=False))


if __name__ == "__main__":
    main()
