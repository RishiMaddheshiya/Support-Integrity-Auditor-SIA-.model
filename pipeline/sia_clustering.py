"""
SIA Stage 1b - Clustering + Pseudo-Label Fusion (NO sentence-transformers)
===========================================================================
Uses TF-IDF + KMeans instead of sentence-transformers to avoid
dependency conflicts with torch 2.x and transformers versions.

Output: pseudo_labeled_tickets.csv
"""

import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("sia_clustering.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("SIA-Cluster")
SEP = "=" * 60

CLEANED_CSV    = "cleaned_tickets.csv"
LLM_SCORES_CSV = "llm_scores.csv"
OUTPUT_CSV     = "pseudo_labeled_tickets.csv"

LABEL_MAP = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
NUM_MAP   = {v: k for k, v in LABEL_MAP.items()}
N_CLUSTERS = 4


def find_col(df, *keywords):
    for col in df.columns:
        if all(kw.lower() in col.lower() for kw in keywords):
            return col
    return None


def main():
    log.info(SEP)
    log.info("SIA Stage 1b - TF-IDF Clustering + Pseudo-Label Fusion")
    log.info("(No sentence-transformers required)")
    log.info(SEP)

    # ── STEP 1: Load ─────────────────────────────────────────────────
    log.info("STEP 1 - LOAD DATA")
    if not Path(CLEANED_CSV).exists():
        log.error(f"'{CLEANED_CSV}' not found. Run sia_eda.py first.")
        sys.exit(1)
    if not Path(LLM_SCORES_CSV).exists():
        log.error(f"'{LLM_SCORES_CSV}' not found. Run sia_llm_scoring.py first.")
        sys.exit(1)

    clean_df = pd.read_csv(CLEANED_CSV)
    llm_df   = pd.read_csv(LLM_SCORES_CSV)
    log.info(f"  Cleaned : {len(clean_df):,} rows")
    log.info(f"  LLM     : {len(llm_df):,} rows")

    col_subj = find_col(clean_df, "subject")
    col_desc = find_col(clean_df, "description")
    col_prio = find_col(clean_df, "priority") or find_col(clean_df, "level")
    col_rt   = (find_col(clean_df, "resolution", "minute")
                or find_col(clean_df, "resolution", "hour")
                or find_col(clean_df, "resolution", "time"))

    log.info(f"  Subject  : {col_subj}")
    log.info(f"  Desc     : {col_desc}")
    log.info(f"  Priority : {col_prio}")
    log.info(f"  ResTime  : {col_rt}")

    if not col_desc or not col_prio:
        log.error("Missing required columns.")
        sys.exit(1)

    # ── Merge LLM scores ─────────────────────────────────────────────
    if "llm_severity_score" in llm_df.columns:
        if "row_index" in llm_df.columns:
            llm_map = llm_df.set_index("row_index")["llm_severity_score"]
            clean_df["llm_severity_score"] = clean_df.index.map(llm_map).fillna("Medium")
        else:
            clean_df["llm_severity_score"] = llm_df["llm_severity_score"].values
    else:
        log.error("'llm_severity_score' column not found in llm_scores.csv")
        sys.exit(1)

    log.info(f"  LLM dist: {clean_df['llm_severity_score'].value_counts().to_dict()}")

    # ── STEP 2: TF-IDF Vectorization ─────────────────────────────────
    log.info(SEP)
    log.info("STEP 2 - TF-IDF VECTORIZATION + SVD (LSA)")

    texts = (
        clean_df[col_subj].fillna("").astype(str) + " " +
        clean_df[col_desc].fillna("").astype(str)
    ).tolist()

    # TF-IDF → SVD (Latent Semantic Analysis) → L2 normalize
    # This gives semantic-like dense vectors without heavy models
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    svd      = TruncatedSVD(n_components=100, random_state=42)
    norm     = Normalizer(copy=False)
    pipeline = make_pipeline(tfidf, svd, norm)

    log.info(f"  Fitting TF-IDF + SVD on {len(texts):,} tickets...")
    X = pipeline.fit_transform(texts)
    log.info(f"  Feature matrix shape: {X.shape}")

    # ── STEP 3: KMeans Clustering ─────────────────────────────────────
    log.info(SEP)
    log.info("STEP 3 - KMEANS CLUSTERING (k=4)")

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10, max_iter=300)
    cluster_labels = kmeans.fit_predict(X)
    clean_df["cluster_id"] = cluster_labels

    # Order clusters by average resolution time (highest = most severe)
    if col_rt:
        rt_vals = pd.to_numeric(clean_df[col_rt], errors="coerce")
        if rt_vals.isna().sum() > len(clean_df) * 0.5:
            rt_vals = clean_df[col_rt].apply(
                lambda x: float(str(x).split()[0]) if str(x)[0:1].isdigit() else 30.0
            )
        rt_vals = rt_vals.fillna(rt_vals.median())
        clean_df["_rt"] = rt_vals
        order_key = clean_df.groupby("cluster_id")["_rt"].mean().sort_values()
    else:
        # Fallback: order by cluster size (smaller cluster = more extreme = critical)
        sizes = clean_df.groupby("cluster_id").size().sort_values(ascending=False)
        order_key = sizes

    severity_rank = {
        order_key.index[0]: "Low",
        order_key.index[1]: "Medium",
        order_key.index[2]: "High",
        order_key.index[3]: "Critical",
    }
    clean_df["cluster_severity"] = clean_df["cluster_id"].map(severity_rank)

    log.info("  Cluster -> Severity:")
    for cid, sev in sorted(severity_rank.items()):
        n = (clean_df["cluster_id"] == cid).sum()
        log.info(f"    Cluster {cid} -> {sev:<10}  ({n:,} tickets)")

    # ── STEP 4: Signal Agreement ──────────────────────────────────────
    log.info(SEP)
    log.info("STEP 4 - SIGNAL AGREEMENT")

    llm_num     = clean_df["llm_severity_score"].map(LABEL_MAP).fillna(2)
    cluster_num = clean_df["cluster_severity"].map(LABEL_MAP).fillna(2)
    agreement   = (llm_num == cluster_num).mean() * 100
    log.info(f"  LLM vs Cluster agreement: {agreement:.1f}%")

    # ── STEP 5: Fusion ────────────────────────────────────────────────
    log.info(SEP)
    log.info("STEP 5 - FUSION (ceiling of average, LLM tiebreaker)")

    def fuse(llm_lbl, cluster_lbl):
        l = LABEL_MAP.get(str(llm_lbl).strip().title(), 2)
        c = LABEL_MAP.get(str(cluster_lbl).strip().title(), 2)
        fused = int(np.ceil((l + c) / 2))
        return NUM_MAP.get(min(fused, 4), "Medium")

    clean_df["fused_severity"] = [
        fuse(r["llm_severity_score"], r["cluster_severity"])
        for _, r in clean_df.iterrows()
    ]

    log.info("  Fused distribution:")
    for lbl in ["Critical", "High", "Medium", "Low"]:
        cnt = (clean_df["fused_severity"] == lbl).sum()
        log.info(f"    {lbl:<10}: {cnt:>6,}  ({cnt/len(clean_df)*100:.1f}%)")

    # ── STEP 6: Binary Mismatch Label ────────────────────────────────
    log.info(SEP)
    log.info("STEP 6 - BINARY MISMATCH LABEL")

    assigned_num = clean_df[col_prio].apply(
        lambda x: LABEL_MAP.get(str(x).strip().title(), 2)
    )
    fused_num = clean_df["fused_severity"].map(LABEL_MAP).fillna(2)

    clean_df["assigned_num"]   = assigned_num
    clean_df["fused_num"]      = fused_num
    clean_df["severity_delta"] = fused_num - assigned_num
    clean_df["mismatch"]       = ((fused_num - assigned_num).abs() >= 2).astype(int)

    n_mm = clean_df["mismatch"].sum()
    n_ok = len(clean_df) - n_mm
    log.info(f"  Consistent (0): {n_ok:>6,}  ({n_ok/len(clean_df)*100:.1f}%)")
    log.info(f"  Mismatch   (1): {n_mm:>6,}  ({n_mm/len(clean_df)*100:.1f}%)")

    # If zero mismatches, lower threshold to 1
    if n_mm == 0:
        log.warning("  Zero mismatches with threshold=2. Lowering to threshold=1")
        clean_df["mismatch"] = ((fused_num - assigned_num).abs() >= 1).astype(int)
        n_mm = clean_df["mismatch"].sum()
        log.info(f"  After adjustment - Mismatch: {n_mm:,}")

    # ── STEP 7: Save ─────────────────────────────────────────────────
    if "_rt" in clean_df.columns:
        clean_df = clean_df.drop(columns=["_rt"])

    clean_df.to_csv(OUTPUT_CSV, index=False)
    log.info(SEP)
    log.info(f"Saved {len(clean_df):,} rows -> '{OUTPUT_CSV}'")
    log.info("New columns: llm_severity_score, cluster_id, cluster_severity,")
    log.info("             fused_severity, assigned_num, fused_num,")
    log.info("             severity_delta, mismatch")
    log.info("Stage 1b COMPLETE")


if __name__ == "__main__":
    main()
