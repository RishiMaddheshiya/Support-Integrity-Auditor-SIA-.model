"""
Support Integrity Auditor (SIA) -- Standalone Inference Script
=============================================================
Accepts a CSV of support tickets and outputs:
  1. predictions.csv  -- per-row predictions with confidence + severity delta
  2. evidence_dossiers_predicted.json  -- full JSON dossier for every mismatch

Usage
-----
  python predict.py --input tickets.csv
  python predict.py --input tickets.csv --output results.csv --threshold 0.45
  python predict.py --input tickets.csv --model_dir ./saved_model

Arguments
---------
  --input        Path to input CSV  (required)
  --output       Path for output CSV  (default: predictions.csv)
  --dossier_out  Path for dossier JSON  (default: evidence_dossiers_predicted.json)
  --model_dir    Path to saved LoRA model  (default: ./saved_model)
  --tok_dir      Path to saved tokenizer  (default: ./saved_tokenizer)
  --threshold    Mismatch probability threshold  (default: 0.50)
  --batch_size   Inference batch size  (default: 32)

Expected CSV columns (flexible name matching)
---------------------------------------------
  Ticket Subject        -> any column containing "subject"
  Ticket Description    -> any column containing "description"
  Assigned Priority     -> any column containing "priority" or "assigned"
  Ticket Channel        -> any column containing "channel"          [optional]
  Ticket Type           -> any column containing "type"             [optional]
  Resolution Time       -> any column containing "resolution"/"time" [optional]
"""
import sys, io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


import os
import re
import sys
import json
import logging
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# 0.  LOGGING
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("sia_predict.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("SIA-Predict")

SEP = "=" * 68


# ---------------------------------------------------------------------
# 1.  CLI ARGUMENTS
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="SIA -- Inference on new ticket CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",        required=True,  help="Path to input tickets CSV.")
    p.add_argument("--output",       default="predictions.csv",
                   help="Path for output predictions CSV.")
    p.add_argument("--dossier_out",  default="evidence_dossiers_predicted.json",
                   help="Path for output Evidence Dossier JSON.")
    p.add_argument("--model_dir",    default="./saved_model",
                   help="Path to saved LoRA adapter directory.")
    p.add_argument("--tok_dir",      default="./saved_tokenizer",
                   help="Path to saved tokenizer directory.")
    p.add_argument("--threshold",    type=float, default=0.50,
                   help="Probability threshold for mismatch classification.")
    p.add_argument("--batch_size",   type=int,   default=32,
                   help="Inference batch size.")
    return p.parse_args()


# ---------------------------------------------------------------------
# 2.  CONSTANTS
# ---------------------------------------------------------------------
LABEL_TO_NUM = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
NUM_TO_LABEL = {v: k for k, v in LABEL_TO_NUM.items()}
MAX_SEQ_LEN  = 256
BASE_MODEL   = "microsoft/deberta-v3-small"

# Corpus reference percentiles for resolution-time interpretation
RT_P25, RT_P50, RT_P75 = 30.0, 60.0, 120.0

ESCALATION_CATALOGUE = [
    ("data loss",      3.0), ("outage",   3.0), ("emergency",     3.0),
    ("cannot access",  2.8), ("failure",  2.5), ("not working",   2.5),
    ("critical",       2.5), ("sla",      2.3), ("escalate",      2.3),
    ("broken",         2.0), ("down",     2.0), ("urgent",        2.0),
    ("immediately",    1.8),
]
NEGATION_WORDS = ["not", "never", "no", "unable", "can't", "won't", "doesn't"]


# ---------------------------------------------------------------------
# 3.  MODEL LOADING
# ---------------------------------------------------------------------
def load_model(model_dir: str, tok_dir: str):
    """Load the LoRA-adapted DeBERTa model and tokenizer."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from peft import PeftModel
    except ImportError as e:
        log.error(f"Missing dependency: {e}. Run: pip install transformers peft torch")
        sys.exit(1)

    log.info(f"Loading tokenizer from '{tok_dir}' …")
    if not Path(tok_dir).exists():
        log.error(
            f"Tokenizer directory '{tok_dir}' not found.\n"
            "Run 'python train_pipeline.py' first to train and save the model."
        )
        sys.exit(1)

    log.info(f"Loading LoRA model from '{model_dir}' …")
    if not Path(model_dir).exists():
        log.error(
            f"Model directory '{model_dir}' not found.\n"
            "Run 'python train_pipeline.py' first to train and save the model."
        )
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {str(device).upper()}")

    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    base      = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2, ignore_mismatched_sizes=True
    )
    model = PeftModel.from_pretrained(base, model_dir)
    model.to(device).eval()

    log.info("[OK]  Model + tokenizer loaded successfully.")
    return tokenizer, model, device


# ---------------------------------------------------------------------
# 4.  COLUMN RESOLVER
# ---------------------------------------------------------------------
def find_col(df: pd.DataFrame, *keywords) -> str | None:
    """Return first column whose name contains ALL keywords (case-insensitive)."""
    for col in df.columns:
        cl = col.lower()
        if all(kw.lower() in cl for kw in keywords):
            return col
    return None


def resolve_columns(df: pd.DataFrame) -> dict:
    """Map semantic field names to actual DataFrame column names."""
    mapping = {
        "subject"    : find_col(df, "subject"),
        "description": find_col(df, "description"),
        "priority"   : find_col(df, "priority") or find_col(df, "assigned"),
        "channel"    : find_col(df, "channel"),
        "type"       : find_col(df, "type"),
        "res_time"   : find_col(df, "resolution", "minute")
                       or find_col(df, "resolution", "time")
                       or find_col(df, "resolution"),
    }
    missing = [k for k, v in mapping.items() if v is None and k in ("subject", "description", "priority")]
    if missing:
        log.error(
            f"Required columns missing from CSV: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )
        sys.exit(1)
    log.info("Column mapping resolved:")
    for k, v in mapping.items():
        log.info(f"  {k:<14} -> {v}")
    return mapping


# ---------------------------------------------------------------------
# 5.  NLP SIGNAL HELPERS
# ---------------------------------------------------------------------
def extract_top_keyword(text: str):
    if not text:
        return None, 0.0
    tl = text.lower()
    found = [
        (phrase, weight)
        for phrase, weight in ESCALATION_CATALOGUE
        if re.search(r"\b" + re.escape(phrase) + r"\b", tl)
    ]
    found.sort(key=lambda x: x[1], reverse=True)
    return (found[0][0], found[0][1]) if found else (None, 0.0)


def compute_keyword_density(subject: str, description: str):
    text  = f"{subject} {description}".strip()
    words = text.split()
    total = max(len(words), 1)
    esc   = sum(
        len(re.findall(r"\b" + re.escape(p) + r"\b", text.lower()))
        for p, _ in ESCALATION_CATALOGUE
    )
    neg   = sum(
        len(re.findall(r"\b" + re.escape(w) + r"\b", text.lower()))
        for w in NEGATION_WORDS
    )
    return round((esc * 2 + neg) / total, 4), esc, neg


def density_to_label(d: float) -> str:
    if d > 0.08: return "Critical"
    if d > 0.05: return "High"
    if d > 0.02: return "Medium"
    return "Low"


def interpret_resolution_time(rt):
    try:
        rt = float(rt)
    except (TypeError, ValueError):
        return "Unknown", "Resolution time data unavailable."
    if rt > RT_P75:
        return "Critical", (f"{rt:.1f} min -- top quartile (P75={RT_P75:.0f} min), "
                             "indicative of a complex or escalated case.")
    elif rt > RT_P50:
        return "High",     (f"{rt:.1f} min -- above the median ({RT_P50:.0f} min), "
                             "above-average resolution effort.")
    elif rt > RT_P25:
        return "Medium",   (f"{rt:.1f} min -- second quartile "
                             f"(P25={RT_P25:.0f}--P50={RT_P50:.0f} min), moderate complexity.")
    else:
        return "Low",      (f"{rt:.1f} min -- bottom quartile (≤P25={RT_P25:.0f} min), "
                             "straightforward resolution.")


# ---------------------------------------------------------------------
# 6.  INPUT TEXT BUILDER
# ---------------------------------------------------------------------
def build_input_text(subject, description, channel, resolution_time) -> str:
    desc_trunc = str(description)[:400]
    return (
        f"[SUBJECT] {subject} "
        f"[DESC] {desc_trunc} "
        f"[CHANNEL] {channel} "
        f"[RESTIME] {resolution_time}"
    )


# ---------------------------------------------------------------------
# 7.  BATCH INFERENCE
# ---------------------------------------------------------------------
def run_batch_inference(tokenizer, model, device, texts: list, batch_size: int,
                        threshold: float):
    """
    Returns (predictions: list[int], probabilities: list[float]).
    predictions[i] = 1 if prob_mismatch[i] >= threshold else 0
    """
    import torch

    all_probs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc   = tokenizer(
            batch,
            max_length=MAX_SEQ_LEN,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            out   = model(**{k: v.to(device) for k, v in enc.items()})
            probs = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
        all_probs.extend(probs.tolist())

        processed = min(start + batch_size, len(texts))
        if processed % 100 == 0 or processed == len(texts):
            log.info(f"  Inference progress: {processed}/{len(texts)} rows")

    preds = [1 if p >= threshold else 0 for p in all_probs]
    return preds, all_probs


# ---------------------------------------------------------------------
# 8.  DOSSIER GENERATOR
# ---------------------------------------------------------------------
def generate_dossier(
    ticket_id, subject, description, channel, resolution_time,
    assigned_priority, inferred_severity, predicted_label, prob_mismatch,
) -> dict:
    pri_num = LABEL_TO_NUM.get(str(assigned_priority).strip().title(), 2)
    inf_num = LABEL_TO_NUM.get(str(inferred_severity).strip().title(), 2)
    delta   = inf_num - pri_num

    if delta > 0:
        mismatch_type = "Hidden Crisis"
    elif delta < 0:
        mismatch_type = "False Alarm"
    else:
        mismatch_type = "Hidden Crisis" if prob_mismatch >= 0.5 else "False Alarm"

    top_kw, kw_wt       = extract_top_keyword(f"{subject} {description}")
    density, esc, neg   = compute_keyword_density(str(subject), str(description))
    rt_lbl, rt_interp   = interpret_resolution_time(resolution_time)
    rule_lbl             = density_to_label(density)

    desc_snip = str(description)[:120] + "…" if len(str(description)) > 120 else str(description)
    subj_snip = str(subject)[:80]

    if predicted_label == 1:
        if mismatch_type == "Hidden Crisis":
            analysis = (
                f"The ticket '{subj_snip}' describes a {inferred_severity}-severity issue "
                f"yet was assigned {assigned_priority} -- a gap of {abs(delta)} severity "
                f"level(s). "
                f"{'Escalation keyword ' + repr(top_kw) + ' found in ticket text. ' if top_kw else 'No standard escalation keywords detected -- polite framing may obscure true impact. '}"
                f"Resolution time ({resolution_time} min, {rt_lbl} tier) corroborates "
                f"severity inference. Excerpt: \"{desc_snip}\""
            )
        else:
            analysis = (
                f"The ticket '{subj_snip}' was assigned {assigned_priority} priority, "
                f"but multi-signal analysis infers {inferred_severity} -- "
                f"{abs(delta)} level(s) lower. "
                f"Keyword density ({density:.4f}) {'is elevated but ' if density > 0.05 else 'is low and '}"
                f"the described issue does not constitute {assigned_priority}-tier impact. "
                f"Resolution time ({resolution_time} min, {rt_lbl} tier) is consistent "
                f"with {inferred_severity} severity. Excerpt: \"{desc_snip}\""
            )
    else:
        analysis = (
            f"Signals are consistent: '{subj_snip}' aligns with {assigned_priority} "
            f"priority. Keyword density ({density:.4f}) and resolution time "
            f"({resolution_time} min, {rt_lbl} tier) both support the assigned severity level."
        )

    return {
        "ticket_id"         : str(ticket_id),
        "assigned_priority" : str(assigned_priority),
        "inferred_severity" : str(inferred_severity),
        "mismatch_type"     : mismatch_type if predicted_label == 1 else "None",
        "severity_delta"    : delta,
        "feature_evidence"  : [
            {
                "signal" : "keyword",
                "value"  : top_kw or "[none detected]",
                "weight" : f"{kw_wt:.1f}",
            },
            {
                "signal"        : "resolution_time",
                "value"         : f"{resolution_time} min",
                "interpretation": rt_interp,
            },
            {
                "signal" : "rule_nlp",
                "value"  : f"{rule_lbl} (density={density:.4f})",
                "weight" : f"{LABEL_TO_NUM.get(rule_lbl, 1):.1f}",
            },
            {
                "signal" : "classifier",
                "value"  : f"mismatch={predicted_label}",
                "weight" : f"{prob_mismatch * 100:.1f}%",
            },
        ],
        "constraint_analysis": analysis,
        "confidence"         : f"{prob_mismatch * 100:.1f}%",
    }


# ---------------------------------------------------------------------
# 9.  MAIN
# ---------------------------------------------------------------------
def main():
    args = parse_args()

    log.info(SEP)
    log.info("  Support Integrity Auditor (SIA) -- Inference Script")
    log.info(SEP)
    log.info(f"  Input CSV       : {args.input}")
    log.info(f"  Output CSV      : {args.output}")
    log.info(f"  Dossier JSON    : {args.dossier_out}")
    log.info(f"  Model directory : {args.model_dir}")
    log.info(f"  Threshold       : {args.threshold}")
    log.info(f"  Batch size      : {args.batch_size}")
    log.info("")

    # -- 1. Load input CSV --------------------------------------------
    if not Path(args.input).exists():
        log.error(f"Input file not found: '{args.input}'")
        sys.exit(1)

    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        log.error(f"Failed to read CSV: {e}")
        sys.exit(1)

    log.info(f"[OK]  Loaded {len(df):,} rows × {df.shape[1]} columns from '{args.input}'")

    # -- 2. Resolve columns -------------------------------------------
    cols = resolve_columns(df)

    # -- 3. Load model ------------------------------------------------
    tokenizer, model, device = load_model(args.model_dir, args.tok_dir)

    # -- 4. Prepare inputs --------------------------------------------
    log.info("\nBuilding input texts …")
    input_texts = []
    rows_meta   = []   # store cleaned field values for dossier building

    for idx, row in df.iterrows():
        subject  = str(row.get(cols["subject"],     "")).strip()
        desc     = str(row.get(cols["description"], "")).strip()
        channel  = str(row.get(cols["channel"],     "Email")).strip() if cols["channel"] else "Email"
        prio_raw = str(row.get(cols["priority"],    "Medium")).strip().title()
        prio     = prio_raw if prio_raw in LABEL_TO_NUM else "Medium"

        rt_raw = row.get(cols["res_time"], 60.0) if cols["res_time"] else 60.0
        try:
            rt = float(rt_raw)
        except (TypeError, ValueError):
            rt = 60.0

        ticket_id = (
            str(row.get("Ticket ID", ""))
            or str(row.get("ticket_id", ""))
            or str(row.get("id", ""))
            or f"ROW-{idx:04d}"
        )

        input_texts.append(build_input_text(subject, desc, channel, rt))
        rows_meta.append({
            "ticket_id" : ticket_id,
            "subject"   : subject,
            "desc"      : desc,
            "channel"   : channel,
            "rt"        : rt,
            "priority"  : prio,
        })

    # -- 5. Batch inference -------------------------------------------
    log.info(f"\nRunning inference on {len(input_texts):,} tickets …")
    preds, probs = run_batch_inference(
        tokenizer, model, device, input_texts,
        args.batch_size, args.threshold
    )
    log.info("[OK]  Inference complete.")

    # -- 6. Derive inferred severity + dossiers -----------------------
    log.info("\nGenerating evidence dossiers for mismatch tickets …")
    dossiers     = []
    result_rows  = []

    for i, meta in enumerate(rows_meta):
        pred        = preds[i]
        prob        = probs[i]
        density, _, _ = compute_keyword_density(meta["subject"], meta["desc"])
        rule_sev    = density_to_label(density)

        inferred    = rule_sev if pred == 0 else (
            "Critical" if prob > 0.80 else "High" if prob > 0.65 else "Medium"
        )

        pri_num = LABEL_TO_NUM.get(meta["priority"], 2)
        inf_num = LABEL_TO_NUM.get(inferred, 2)
        delta   = inf_num - pri_num

        mt = "None"
        if pred == 1:
            mt = "Hidden Crisis" if delta > 0 else ("False Alarm" if delta < 0
                 else ("Hidden Crisis" if prob >= 0.5 else "False Alarm"))

        result_rows.append({
            "ticket_id"          : meta["ticket_id"],
            "assigned_priority"  : meta["priority"],
            "inferred_severity"  : inferred,
            "predicted_label"    : pred,
            "mismatch_type"      : mt,
            "confidence"         : f"{prob * 100:.1f}%",
            "prob_mismatch"      : round(prob, 4),
            "severity_delta"     : delta,
            "top_keyword"        : (extract_top_keyword(
                                        f"{meta['subject']} {meta['desc']}"
                                    ))[0] or "",
            "keyword_density"    : density,
            "resolution_time_min": meta["rt"],
        })

        if pred == 1:
            d = generate_dossier(
                ticket_id        = meta["ticket_id"],
                subject          = meta["subject"],
                description      = meta["desc"],
                channel          = meta["channel"],
                resolution_time  = meta["rt"],
                assigned_priority= meta["priority"],
                inferred_severity= inferred,
                predicted_label  = pred,
                prob_mismatch    = prob,
            )
            dossiers.append(d)

    # -- 7. Save predictions CSV --------------------------------------
    results_df = pd.DataFrame(result_rows)
    results_df.to_csv(args.output, index=False)
    log.info(f"[OK]  Predictions saved -> '{args.output}'  ({len(results_df):,} rows)")

    # -- 8. Save dossiers JSON ----------------------------------------
    dossier_payload = {
        "total_tickets"      : len(results_df),
        "mismatch_count"     : int(results_df["predicted_label"].sum()),
        "hidden_crisis_count": int((results_df["mismatch_type"] == "Hidden Crisis").sum()),
        "false_alarm_count"  : int((results_df["mismatch_type"] == "False Alarm").sum()),
        "threshold_used"     : args.threshold,
        "dossiers"           : dossiers,
    }
    with open(args.dossier_out, "w", encoding="utf-8") as f:
        json.dump(dossier_payload, f, indent=2, ensure_ascii=False)
    log.info(f"[OK]  Dossiers saved   -> '{args.dossier_out}'  ({len(dossiers)} mismatch dossiers)")

    # -- 9. Summary report --------------------------------------------
    total       = len(results_df)
    n_mismatch  = int(results_df["predicted_label"].sum())
    n_ok        = total - n_mismatch
    n_hc        = int((results_df["mismatch_type"] == "Hidden Crisis").sum())
    n_fa        = int((results_df["mismatch_type"] == "False Alarm").sum())

    log.info("")
    log.info(SEP)
    log.info("  -- Prediction Summary -------------------------------------")
    log.info(f"  Total tickets      : {total:>6,}")
    log.info(f"  Consistent (OK)    : {n_ok:>6,}  ({n_ok/total*100:.1f}%)")
    log.info(f"  Mismatch flagged   : {n_mismatch:>6,}  ({n_mismatch/total*100:.1f}%)")
    log.info(f"    +- Hidden Crisis : {n_hc:>6,}  (under-prioritised)")
    log.info(f"    +- False Alarm   : {n_fa:>6,}  (over-prioritised)")
    log.info("")
    log.info(f"  Output CSV         : {args.output}")
    log.info(f"  Evidence Dossiers  : {args.dossier_out}")
    log.info(SEP)


if __name__ == "__main__":
    main()
