"""
SIA Stage 1a - Fast LLM-Style Severity Scoring (FIXED)
No LLM download needed. Runs in ~30 seconds.
Output: llm_scores.csv
"""

import re
import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("sia_llm_scoring.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("SIA-LLM")
SEP = "=" * 60

INPUT_FILE  = "cleaned_tickets.csv"
OUTPUT_FILE = "llm_scores.csv"

# Severity keywords with weights
CRITICAL_KW = [
    "data loss", "data breach", "security breach", "system down",
    "production down", "ransomware", "hacked", "complete outage",
    "cannot access", "service unavailable", "data corruption",
]
HIGH_KW = [
    "not working", "failure", "emergency", "broken", "outage",
    "urgent", "escalate", "sla", "major issue", "access denied",
    "payment failed", "billing error", "cannot login", "crash",
    "crashing", "crashes",
]
MEDIUM_KW = [
    "error", "failed", "incorrect", "wrong", "missing",
    "problem", "issue", "bug", "not loading", "spinning",
    "slow", "delay", "not syncing", "update", "upgrade",
]
LOW_KW = [
    "question", "inquiry", "how do i", "how to", "where is",
    "information", "request", "feedback", "feature", "suggestion",
    "status", "update me", "let me know",
]

NEGATIONS = ["not", "never", "no", "unable", "cannot", "can't",
             "won't", "doesn't", "didn't", "don't"]

LABEL_MAP = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}


def find_col(df, *keywords):
    for col in df.columns:
        if all(kw.lower() in col.lower() for kw in keywords):
            return col
    return None


def classify_ticket(subject: str, description: str) -> str:
    """
    Rule-based severity classification.
    Returns: Critical / High / Medium / Low
    """
    text = f"{subject} {description}".lower()

    # Count keyword hits per tier
    crit_hits   = sum(1 for kw in CRITICAL_KW if re.search(r"\b" + re.escape(kw) + r"\b", text))
    high_hits   = sum(1 for kw in HIGH_KW    if re.search(r"\b" + re.escape(kw) + r"\b", text))
    medium_hits = sum(1 for kw in MEDIUM_KW  if re.search(r"\b" + re.escape(kw) + r"\b", text))
    low_hits    = sum(1 for kw in LOW_KW     if re.search(r"\b" + re.escape(kw) + r"\b", text))

    neg_count = sum(len(re.findall(r"\b" + re.escape(w) + r"\b", text)) for w in NEGATIONS)
    neg_boost = neg_count > 0  # negation pushes severity up one level

    # Decision logic
    if crit_hits >= 1:
        return "Critical"

    if high_hits >= 2:
        return "Critical" if neg_boost else "High"

    if high_hits == 1:
        return "High"

    if medium_hits >= 2:
        return "High" if neg_boost else "Medium"

    if medium_hits == 1:
        return "Medium" if not neg_boost else "High"

    if low_hits >= 1:
        return "Low"

    # Fallback: use description length as proxy
    desc_len = len(description.split())
    if desc_len > 80:   return "High"
    if desc_len > 40:   return "Medium"
    if desc_len > 15:   return "Medium"
    return "Low"


def main():
    log.info(SEP)
    log.info("SIA Stage 1a - Fast Severity Scoring (FIXED VERSION)")
    log.info(SEP)

    if not Path(INPUT_FILE).exists():
        log.error(f"'{INPUT_FILE}' not found. Run sia_eda.py first.")
        sys.exit(1)

    df = pd.read_csv(INPUT_FILE)
    log.info(f"Loaded {len(df):,} rows from '{INPUT_FILE}'")

    col_subj = find_col(df, "subject")
    col_desc = find_col(df, "description")
    col_prio = find_col(df, "priority") or find_col(df, "level")

    log.info(f"  Subject  : {col_subj}")
    log.info(f"  Desc     : {col_desc}")
    log.info(f"  Priority : {col_prio}")

    if not col_subj or not col_desc:
        log.error("Could not find Subject or Description columns.")
        sys.exit(1)

    subjects = df[col_subj].fillna("").astype(str)
    descs    = df[col_desc].fillna("").astype(str)

    log.info("Classifying tickets...")
    labels = []
    for i in range(len(df)):
        lbl = classify_ticket(subjects.iloc[i], descs.iloc[i])
        labels.append(lbl)
        if (i + 1) % 5000 == 0:
            log.info(f"  {i+1:,} / {len(df):,} done...")

    df_out = df.copy()
    df_out["llm_severity_score"] = labels
    df_out["llm_severity_num"]   = [LABEL_MAP.get(l, 2) for l in labels]
    # Add row_index so sia_clustering.py can merge on it
    df_out["row_index"] = range(len(df_out))

    log.info("Distribution:")
    for lbl in ["Critical", "High", "Medium", "Low"]:
        cnt = labels.count(lbl)
        log.info(f"  {lbl:<10}: {cnt:>6,}  ({cnt/len(df)*100:.1f}%)")

    if col_prio:
        assigned_num = df[col_prio].apply(lambda x: LABEL_MAP.get(str(x).strip().title(), 2))
        inferred_num = df_out["llm_severity_num"]
        agreement    = (assigned_num == inferred_num).mean() * 100
        log.info(f"Agreement with assigned priority: {agreement:.1f}%")

    df_out.to_csv(OUTPUT_FILE, index=False)
    log.info(f"Saved {len(df_out):,} rows -> '{OUTPUT_FILE}'")
    log.info("Stage 1a COMPLETE")


if __name__ == "__main__":
    main()
