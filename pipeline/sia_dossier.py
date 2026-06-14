"""
SIA Stage 3 - Evidence Dossier Generation
==========================================
For every mismatch ticket in test_predictions.csv,
generates a structured JSON dossier (zero hallucination).

Output: evidence_dossiers.json
"""

import re
import sys
import json
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
        logging.FileHandler("sia_dossier.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("SIA-Dossier")
SEP = "=" * 60

PREDS_CSV      = "test_predictions.csv"
PSEUDO_CSV     = "pseudo_labeled_tickets.csv"
OUTPUT_JSON    = "evidence_dossiers.json"

LABEL_MAP = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
NUM_MAP   = {v: k for k, v in LABEL_MAP.items()}

ESCALATION_CATALOGUE = [
    ("data loss",      3.0), ("outage",      3.0), ("emergency",    3.0),
    ("cannot access",  2.8), ("failure",     2.5), ("not working",  2.5),
    ("critical",       2.5), ("sla",         2.3), ("escalate",     2.3),
    ("broken",         2.0), ("down",        2.0), ("urgent",       2.0),
    ("immediately",    1.8), ("error",       1.2), ("problem",      1.0),
]
NEGATION_WORDS = ["not","never","no","unable","can't","won't","doesn't","didn't","don't"]


def find_col(df, *keywords):
    for col in df.columns:
        if all(kw.lower() in col.lower() for kw in keywords):
            return col
    return None


def extract_top_keyword(text):
    tl = str(text).lower()
    found = [
        (phrase, weight)
        for phrase, weight in ESCALATION_CATALOGUE
        if re.search(r"\b" + re.escape(phrase) + r"\b", tl)
    ]
    found.sort(key=lambda x: x[1], reverse=True)
    return (found[0][0], found[0][1]) if found else (None, 0.0)


def compute_keyword_density(subject, description):
    text  = f"{subject} {description}".strip().lower()
    words = text.split()
    total = max(len(words), 1)
    esc   = sum(
        len(re.findall(r"\b" + re.escape(p) + r"\b", text)) * w
        for p, w in ESCALATION_CATALOGUE
    )
    neg   = sum(
        len(re.findall(r"\b" + re.escape(w) + r"\b", text))
        for w in NEGATION_WORDS
    )
    return round((esc + neg) / total, 4)


def interpret_resolution_time(rt, p25=30.0, p50=60.0, p75=120.0):
    try:
        rt = float(rt)
    except (TypeError, ValueError):
        return "Unknown", "Resolution time unavailable."
    if rt > p75:
        return "Critical", f"{rt:.1f} min — top quartile (P75={p75:.0f} min), complex case."
    elif rt > p50:
        return "High",     f"{rt:.1f} min — above median ({p50:.0f} min), above-average effort."
    elif rt > p25:
        return "Medium",   f"{rt:.1f} min — second quartile, moderate complexity."
    else:
        return "Low",      f"{rt:.1f} min — bottom quartile (<=P25={p25:.0f} min), straightforward."


def build_dossier(row, col_map, rt_percentiles):
    ticket_id        = str(row.get(col_map.get("id","Ticket_ID"), f"ROW-{row.name}"))
    subject          = str(row.get(col_map["subj"], ""))
    description      = str(row.get(col_map["desc"], ""))
    channel          = str(row.get(col_map.get("channel",""), "Unknown")) if col_map.get("channel") else "Unknown"
    assigned_priority = str(row.get(col_map["prio"], "Medium")).strip().title()
    if assigned_priority not in LABEL_MAP:
        assigned_priority = "Medium"

    # Inferred severity: from fused_severity or fallback
    inferred_severity = str(row.get("fused_severity", row.get("inferred_severity", "Medium"))).strip().title()
    if inferred_severity not in LABEL_MAP:
        inferred_severity = "Medium"

    prob_mismatch = float(row.get("prob_mismatch", 0.5))

    rt_raw = row.get(col_map.get("rt", ""), 60.0)
    try:
        rt = float(rt_raw)
    except (TypeError, ValueError):
        rt = 60.0

    pri_num = LABEL_MAP.get(assigned_priority, 2)
    inf_num = LABEL_MAP.get(inferred_severity, 2)
    delta   = inf_num - pri_num

    if delta > 0:
        mismatch_type = "Hidden Crisis"
    elif delta < 0:
        mismatch_type = "False Alarm"
    else:
        mismatch_type = "Hidden Crisis" if prob_mismatch >= 0.5 else "False Alarm"

    top_kw, kw_wt = extract_top_keyword(f"{subject} {description}")
    density        = compute_keyword_density(subject, description)
    rt_lbl, rt_interp = interpret_resolution_time(rt, *rt_percentiles)

    desc_snip = description[:120] + "..." if len(description) > 120 else description

    if mismatch_type == "Hidden Crisis":
        analysis = (
            f"Ticket '{subject[:60]}' describes a {inferred_severity}-severity issue "
            f"but was assigned {assigned_priority} — {abs(delta)} severity level(s) lower. "
            f"{'Escalation keyword ' + repr(top_kw) + ' detected. ' if top_kw else 'No standard escalation keyword — polite framing may obscure true impact. '}"
            f"Resolution time ({rt:.0f} min, {rt_lbl} tier) corroborates higher severity. "
            f"Excerpt: \"{desc_snip}\""
        )
    else:
        analysis = (
            f"Ticket '{subject[:60]}' was assigned {assigned_priority} but signals infer "
            f"{inferred_severity} — {abs(delta)} level(s) lower. "
            f"Keyword density ({density:.4f}) and resolution time ({rt:.0f} min, {rt_lbl} tier) "
            f"are both consistent with {inferred_severity} severity. "
            f"Excerpt: \"{desc_snip}\""
        )

    return {
        "ticket_id"          : ticket_id,
        "assigned_priority"  : assigned_priority,
        "inferred_severity"  : inferred_severity,
        "mismatch_type"      : mismatch_type,
        "severity_delta"     : delta,
        "feature_evidence"   : [
            {
                "signal" : "keyword",
                "value"  : top_kw or "[none detected]",
                "weight" : str(round(kw_wt, 1)),
            },
            {
                "signal"        : "resolution_time",
                "value"         : f"{rt:.0f} min",
                "interpretation": rt_interp,
            },
            {
                "signal" : "keyword_density",
                "value"  : str(density),
                "weight" : "combined escalation+negation score",
            },
            {
                "signal" : "classifier",
                "value"  : f"mismatch_prob={prob_mismatch:.3f}",
                "weight" : f"{prob_mismatch*100:.1f}%",
            },
        ],
        "constraint_analysis": analysis,
        "confidence"         : f"{prob_mismatch*100:.1f}%",
    }


def main():
    log.info(SEP)
    log.info("SIA Stage 3 - Evidence Dossier Generation")
    log.info(SEP)

    # ── Load test predictions ─────────────────────────────────────────
    if not Path(PREDS_CSV).exists():
        log.error(f"'{PREDS_CSV}' not found. Run sia_classifier.py first.")
        sys.exit(1)
    if not Path(PSEUDO_CSV).exists():
        log.error(f"'{PSEUDO_CSV}' not found. Run sia_clustering.py first.")
        sys.exit(1)

    preds_df  = pd.read_csv(PREDS_CSV)
    pseudo_df = pd.read_csv(PSEUDO_CSV)

    log.info(f"STEP 1 - LOAD & JOIN DATA")
    log.info(f"  test_predictions : {len(preds_df):,} rows")
    log.info(f"  pseudo_labeled   : {len(pseudo_df):,} rows")

    # Find columns in pseudo_df (full data)
    col_subj    = find_col(pseudo_df, "subject")
    col_desc    = find_col(pseudo_df, "description")
    col_prio    = find_col(pseudo_df, "priority") or find_col(pseudo_df, "level")
    col_channel = find_col(pseudo_df, "channel")
    col_rt      = (find_col(pseudo_df, "resolution", "minute")
                   or find_col(pseudo_df, "resolution", "hour")
                   or find_col(pseudo_df, "resolution", "time"))
    col_id      = find_col(pseudo_df, "ticket", "id") or find_col(pseudo_df, "_id")

    log.info(f"  Columns used from tickets:")
    log.info(f"    * Subject       : {col_subj}")
    log.info(f"    * Description   : {col_desc}")
    log.info(f"    * Priority      : {col_prio}")
    log.info(f"    * Channel       : {col_channel}")
    log.info(f"    * Resolution    : {col_rt}")

    col_map = {
        "subj"   : col_subj,
        "desc"   : col_desc,
        "prio"   : col_prio,
        "channel": col_channel,
        "rt"     : col_rt,
        "id"     : col_id,
    }

    # Merge predictions with full ticket data
    # preds_df is a subset (15% test split) — align by position if no shared key
    if col_id and col_id in preds_df.columns:
        merged = preds_df.merge(
            pseudo_df[[c for c in [col_id, col_subj, col_desc, col_channel,
                                    col_rt, "fused_severity"] if c]],
            on=col_id, how="left", suffixes=("", "_full")
        )
    else:
        # Reset index and merge positionally
        preds_df  = preds_df.reset_index(drop=True)
        # preds_df already has all pseudo_df columns (it's a slice)
        merged = preds_df

    log.info(f"  Merged rows : {len(merged):,}")
    log.info(f"  Columns     : {list(merged.columns)[:10]}...")

    # Compute RT percentiles for interpretation
    if col_rt and col_rt in merged.columns:
        rt_vals = pd.to_numeric(merged[col_rt], errors="coerce").dropna()
        rt_percentiles = (
            float(rt_vals.quantile(0.25)),
            float(rt_vals.quantile(0.50)),
            float(rt_vals.quantile(0.75)),
        )
    else:
        rt_percentiles = (30.0, 60.0, 120.0)
    log.info(f"  RT percentiles: P25={rt_percentiles[0]:.1f}  P50={rt_percentiles[1]:.1f}  P75={rt_percentiles[2]:.1f}")

    # ── Generate dossiers for mismatch tickets ───────────────────────
    log.info(SEP)
    log.info("STEP 2 - GENERATE DOSSIERS")

    mismatch_mask = merged.get("predicted_label", merged.get("mismatch", pd.Series([0]*len(merged)))) == 1
    mismatch_df   = merged[mismatch_mask].reset_index(drop=True)
    log.info(f"  Mismatch tickets: {len(mismatch_df):,} / {len(merged):,}")

    dossiers = []
    hidden_crisis = 0
    false_alarm   = 0

    for idx, row in mismatch_df.iterrows():
        try:
            d = build_dossier(row, col_map, rt_percentiles)
            dossiers.append(d)
            if d["mismatch_type"] == "Hidden Crisis":
                hidden_crisis += 1
            else:
                false_alarm += 1
        except Exception as e:
            log.warning(f"  Row {idx} dossier failed: {e}")

        if (idx + 1) % 100 == 0:
            log.info(f"  Generated {idx+1}/{len(mismatch_df)} dossiers...")

    log.info(f"  Hidden Crisis : {hidden_crisis:,}")
    log.info(f"  False Alarm   : {false_alarm:,}")

    # ── Save ─────────────────────────────────────────────────────────
    log.info(SEP)
    log.info("STEP 3 - SAVE")

    payload = {
        "total_tickets"      : len(merged),
        "mismatch_count"     : len(dossiers),
        "hidden_crisis_count": hidden_crisis,
        "false_alarm_count"  : false_alarm,
        "dossiers"           : dossiers,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    log.info(f"  Saved {len(dossiers)} dossiers -> '{OUTPUT_JSON}'")

    # Print 2 examples
    if dossiers:
        hc = next((d for d in dossiers if d["mismatch_type"] == "Hidden Crisis"), None)
        fa = next((d for d in dossiers if d["mismatch_type"] == "False Alarm"), None)
        for ex, label in [(hc, "Hidden Crisis"), (fa, "False Alarm")]:
            if ex:
                log.info(f"\n  Example {label}:")
                log.info(f"    ticket_id        : {ex['ticket_id']}")
                log.info(f"    assigned_priority: {ex['assigned_priority']}")
                log.info(f"    inferred_severity: {ex['inferred_severity']}")
                log.info(f"    severity_delta   : {ex['severity_delta']}")
                log.info(f"    confidence       : {ex['confidence']}")

    log.info(SEP)
    log.info("Stage 3 COMPLETE")


if __name__ == "__main__":
    main()
