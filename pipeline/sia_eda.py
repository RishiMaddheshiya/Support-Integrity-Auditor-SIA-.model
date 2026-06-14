"""
Support Integrity Auditor (SIA) -- Step 1: EDA & Data Cleaning
"""

import pandas as pd
import numpy as np
import re
import sys
import io

# ---------------------------------------------
# FIX: Force UTF-8 (prevents UnicodeEncodeError)
# ---------------------------------------------
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ---------------------------------------------
# CONFIG
# ---------------------------------------------
RAW_FILE    = "customer_support_tickets.csv"
OUTPUT_FILE = "cleaned_tickets.csv"

SEPARATOR = "=" * 60


# ---------------------------------------------
# SAFE PRINT (no unicode issues)
# ---------------------------------------------
def safe_print(text):
    print(str(text).encode("ascii", errors="ignore").decode())


# ---------------------------------------------
# LOAD DATA
# ---------------------------------------------
safe_print(SEPARATOR)
safe_print("STEP 1 - LOAD DATASET")
safe_print(SEPARATOR)

try:
    df = pd.read_csv(RAW_FILE, encoding="utf-8", engine="python")
except:
    df = pd.read_csv(RAW_FILE, encoding="latin1", engine="python")

safe_print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")
safe_print(f"Columns: {list(df.columns)}\n")


# ---------------------------------------------
# EDA
# ---------------------------------------------
safe_print(SEPARATOR)
safe_print("STEP 2 - EDA")
safe_print(SEPARATOR)

# Sample
safe_print("\nFirst 5 rows:")
safe_print(df.head().to_string())

# Info
safe_print("\nData Info:")
df.info()

# Missing values
safe_print("\nMissing Values:")
missing = df.isnull().sum()
missing = missing[missing > 0]

if missing.empty:
    safe_print("No missing values")
else:
    safe_print(missing.to_string())


# ---------------------------------------------
# CLEANING
# ---------------------------------------------
safe_print(SEPARATOR)
safe_print("STEP 3 - CLEANING")
safe_print(SEPARATOR)

df_clean = df.copy()


# Column finder
def find_col(df, *keywords):
    for col in df.columns:
        if all(k.lower() in col.lower() for k in keywords):
            return col
    return None


col_desc     = find_col(df_clean, "description")
col_priority = find_col(df_clean, "priority")
col_res_time = find_col(df_clean, "resolution", "time")
col_channel  = find_col(df_clean, "channel")
col_type     = find_col(df_clean, "type")

safe_print(f"Description: {col_desc}")
safe_print(f"Priority: {col_priority}")
safe_print(f"Resolution Time: {col_res_time}")


# Drop null rows
if col_desc and col_priority:
    before = len(df_clean)
    df_clean = df_clean.dropna(subset=[col_desc, col_priority])
    safe_print(f"Dropped {before - len(df_clean)} rows")


# Parse resolution time
def parse_time(val):
    if pd.isna(val):
        return np.nan

    val = str(val).lower().strip()

    hms = re.match(r"(\d+):(\d+):(\d+)", val)
    if hms:
        h, m, s = map(int, hms.groups())
        return h * 60 + m + s / 60

    num = re.findall(r"[\d.]+", val)
    if not num:
        return np.nan

    num = float(num[0])

    if "day" in val:
        return num * 1440
    elif "hour" in val:
        return num * 60
    else:
        return num


if col_res_time:
    df_clean["Resolution_Time_Minutes"] = df_clean[col_res_time].apply(parse_time)


# Convert to category
for col in [col_channel, col_type]:
    if col:
        df_clean[col] = df_clean[col].astype("category")


# Strip whitespace
str_cols = df_clean.select_dtypes(include="object").columns
df_clean[str_cols] = df_clean[str_cols].apply(lambda x: x.str.strip())


# Reset index
df_clean.reset_index(drop=True, inplace=True)


# ---------------------------------------------
# SAVE
# ---------------------------------------------
safe_print(SEPARATOR)
safe_print("STEP 4 - SAVE")
safe_print(SEPARATOR)

df_clean.to_csv(OUTPUT_FILE, index=False)

safe_print(f"Saved file: {OUTPUT_FILE}")
safe_print(f"Final shape: {df_clean.shape}")

safe_print("\nDONE - SIA Step 1 Completed Successfully")