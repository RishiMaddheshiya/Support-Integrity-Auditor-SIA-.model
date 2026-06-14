"""
Support Integrity Auditor (SIA) -- Standalone Training Pipeline
===============================================================
Runs the FULL SIA pipeline end-to-end:

  Stage 0 -> EDA & Cleaning          (sia_eda.py logic)
  Stage 1 -> Pseudo-Label Generation  (llm + cluster + regression + rule-NLP)
  Stage 2 -> DeBERTa-v3-small + LoRA  Binary Mismatch Classifier
  Stage 3 -> Evidence Dossier JSON
  Stage 4 -> Adversarial Robustness Test

Usage
-----
  python train_pipeline.py
  python train_pipeline.py --data_path customer_support_tickets.csv
  python train_pipeline.py --data_path data.csv --output_dir ./my_model --epochs 8

Outputs
-------
  cleaned_tickets.csv
  llm_scores.csv
  pseudo_labeled_tickets.csv
  ablation_table.csv
  saved_model/         (LoRA adapter weights)
  saved_tokenizer/     (DeBERTa tokenizer)
  test_predictions.csv
  evidence_dossiers.json
  adversarial_results.json
  stage2_metrics.json
"""

import os
import re
import sys
import json
import logging
import argparse
import warnings
import subprocess
from pathlib import Path

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
        logging.FileHandler("sia_train.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("SIA-Train")

SEP = "=" * 68


# ---------------------------------------------------------------------
# 1.  CLI ARGUMENTS
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="SIA -- Full Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data_path",
        default="customer_support_tickets.csv",
        help="Path to the raw CRM tickets CSV file.",
    )
    p.add_argument(
        "--output_dir",
        default="./saved_model",
        help="Directory to save the trained LoRA model.",
    )
    p.add_argument(
        "--tokenizer_dir",
        default="./saved_tokenizer",
        help="Directory to save the tokenizer.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs for the DeBERTa classifier.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Training batch size.",
    )
    p.add_argument(
        "--skip_llm",
        action="store_true",
        help="Skip LLM scoring step (use if llm_scores.csv already exists).",
    )
    p.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip classifier training (use if saved_model/ already exists).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------
# 2.  STAGE RUNNER -- runs each sub-script in a controlled subprocess
# ---------------------------------------------------------------------
def run_stage(script_name: str, stage_label: str, env_overrides: dict = None):
    """
    Run a pipeline stage script (e.g. sia_eda.py) as a subprocess.
    Streams stdout/stderr to the logger in real time.
    Raises RuntimeError if the script exits non-zero.
    """
    log.info(SEP)
    log.info(f"STARTING  {stage_label}  ->  {script_name}")
    log.info(SEP)

    if not Path(script_name).exists():
        raise FileNotFoundError(
            f"Script '{script_name}' not found in the current directory. "
            "Make sure all SIA scripts are in the same folder as train_pipeline.py."
        )

    env = os.environ.copy()
    if env_overrides:
        env.update({k: str(v) for k, v in env_overrides.items()})

    proc = subprocess.Popen(
        [sys.executable, script_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    for line in proc.stdout:
        log.info(f"  [{script_name}]  {line.rstrip()}")
    proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(
            f"Stage '{stage_label}' FAILED (exit code {proc.returncode}). "
            f"Check sia_train.log for details."
        )
    log.info(f"[OK]  {stage_label} complete.\n")


# ---------------------------------------------------------------------
# 3.  INLINE HELPERS (used when we can't call sub-scripts)
# ---------------------------------------------------------------------
def patch_data_path(script: str, raw_path: str) -> None:
    """
    Temporarily patch RAW_FILE / INPUT_CSV in a script before running.
    We write an env-var; each script checks SIA_DATA_PATH at the top if present.
    (This avoids editing script source files.)
    """
    pass   # We pass the path via env vars below


def validate_file_exists(path: str, label: str) -> None:
    if not Path(path).exists():
        log.error(f"Expected output file '{path}' ({label}) not found after stage run.")
        sys.exit(1)
    log.info(f"[OK]  Verified output: {path}")


# ---------------------------------------------------------------------
# 4.  MAIN PIPELINE
# ---------------------------------------------------------------------
def main():
    args = parse_args()

    log.info(SEP)
    log.info("  Support Integrity Auditor (SIA) -- Full Training Pipeline")
    log.info(SEP)
    log.info(f"  Data path   : {args.data_path}")
    log.info(f"  Model dir   : {args.output_dir}")
    log.info(f"  Epochs      : {args.epochs}")
    log.info(f"  Batch size  : {args.batch_size}")
    log.info(f"  Skip LLM    : {args.skip_llm}")
    log.info(f"  Skip train  : {args.skip_train}")
    log.info("")

    # -- Sanity check: raw data file ----------------------------------
    if not Path(args.data_path).exists():
        log.error(
            f"Raw data file not found: '{args.data_path}'\n"
            "Download from: https://www.kaggle.com/datasets/ajverse/"
            "customer-support-tickets-crm-dataset/data\n"
            "Then run: python train_pipeline.py --data_path <path>"
        )
        sys.exit(1)

    # -- Copy/rename data file if needed -----------------------------
    src = Path(args.data_path)
    if src.name != "customer_support_tickets.csv":
        import shutil
        dest = Path("customer_support_tickets.csv")
        shutil.copy(src, dest)
        log.info(f"  Copied '{src}' -> 'customer_support_tickets.csv'")

    # -- STAGE 0: EDA & Cleaning --------------------------------------
    run_stage("sia_eda.py", "Stage 0 -- EDA & Data Cleaning")
    validate_file_exists("cleaned_tickets.csv", "Cleaned tickets")

    # -- STAGE 1a: LLM Severity Scoring ------------------------------
    if args.skip_llm and Path("llm_scores.csv").exists():
        log.info("⏩  Skipping LLM scoring (llm_scores.csv already exists).")
    else:
        run_stage("sia_llm_scoring.py", "Stage 1a -- LLM Zero-Shot Severity Scoring")
        validate_file_exists("llm_scores.csv", "LLM scores")

    # -- STAGE 1b: Embedding Clustering ------------------------------
    run_stage("sia_clustering.py", "Stage 1b -- Embedding Clustering + Pseudo-Label Fusion")
    validate_file_exists("pseudo_labeled_tickets.csv", "Pseudo-labeled tickets")

    # -- STAGE 1c: Signals 3 & 4 + Ablation --------------------------
    run_stage(
        "sia_signals_3_4.py",
        "Stage 1c -- RT Regression + Rule-NLP Signals + Ablation Table",
    )
    validate_file_exists("ablation_table.csv", "Ablation table")
    validate_file_exists("pseudo_labeled_tickets.csv", "Updated pseudo-labeled tickets")

    # -- STAGE 2: Classifier Training --------------------------------
    if args.skip_train and Path(args.output_dir).exists():
        log.info("⏩  Skipping classifier training (model directory already exists).")
    else:
        # Inject CLI overrides via environment variables
        env_ov = {
            "SIA_EPOCHS"      : args.epochs,
            "SIA_BATCH_SIZE"  : args.batch_size,
            "SIA_MODEL_DIR"   : args.output_dir,
            "SIA_TOKENIZER_DIR": args.tokenizer_dir,
        }
        run_stage(
            "sia_classifier.py",
            "Stage 2 -- DeBERTa-v3-small + LoRA Classifier Training",
            env_overrides=env_ov,
        )
        validate_file_exists("test_predictions.csv", "Test predictions")
        validate_file_exists(f"{args.output_dir}/adapter_config.json", "LoRA adapter")

    # -- STAGE 3: Dossier Generation ----------------------------------
    run_stage("sia_dossier.py", "Stage 3 -- Evidence Dossier Generation")
    validate_file_exists("evidence_dossiers.json", "Evidence dossiers")

    # -- STAGE 4: Adversarial Robustness Test -------------------------
    run_stage("sia_adversarial.py", "Stage 4 -- Adversarial Robustness Test")
    validate_file_exists("adversarial_results.json", "Adversarial results")

    # -- FINAL REPORT -------------------------------------------------
    log.info(SEP)
    log.info("  ✅  SIA Training Pipeline COMPLETE")
    log.info(SEP)

    # Print metrics summary if available
    metrics_path = Path("stage2_metrics.json")
    if metrics_path.exists():
        with open(metrics_path) as f:
            m = json.load(f)
        log.info("")
        log.info("  -- Classifier Metrics ----------------------------------")
        log.info(f"  Accuracy            : {m.get('test_accuracy', '?') * 100:.2f}%  "
                 f"(req ≥ 83%)")
        log.info(f"  Macro F1            : {m.get('test_f1_macro', '?'):.4f}      "
                 f"(req ≥ 0.82)")
        log.info(f"  Recall class=0 (OK) : {m.get('test_recall_class0', '?') * 100:.2f}%  "
                 f"(req ≥ 78%)")
        log.info(f"  Recall class=1 (MM) : {m.get('test_recall_class1', '?') * 100:.2f}%  "
                 f"(req ≥ 78%)")
        status = "✅ ALL PASSED" if m.get("all_thresholds_passed") else "[!]️  SOME FAILED"
        log.info(f"  Threshold Status    : {status}")

    adv_path = Path("adversarial_results.json")
    if adv_path.exists():
        with open(adv_path) as f:
            adv = json.load(f)
        score = adv.get("correctly_detected", "?")
        bonus = adv.get("bonus_achieved", False)
        log.info(f"\n  Adversarial Score   : {score}/10  "
                 f"{'🏆 BONUS ACHIEVED (+10%)' if bonus else '(need ≥7 for bonus)'}")

    log.info("")
    log.info("  Output files produced:")
    files = [
        "cleaned_tickets.csv",
        "llm_scores.csv",
        "pseudo_labeled_tickets.csv",
        "ablation_table.csv",
        "test_predictions.csv",
        "evidence_dossiers.json",
        "adversarial_results.json",
        "stage2_metrics.json",
        args.output_dir + "/",
        args.tokenizer_dir + "/",
    ]
    for f in files:
        exists = Path(f.rstrip("/")).exists()
        icon   = "[OK]" if exists else "[X]"
        log.info(f"  {icon}  {f}")

    log.info("")
    log.info("  Next step -- launch the Streamlit web app:")
    log.info("    streamlit run app.py")
    log.info(SEP)


if __name__ == "__main__":
    main()
