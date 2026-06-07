"""
=============================================================
  Member 1 – Data Preparation Script
  Project : Fuzzy Logic for Earthquake Magnitude Prediction
  Dataset : Japan Earthquakes 2001-2018 (14,092 rows)
  Author  : Member 1 (The Architect)
=============================================================

PURPOSE
-------
Cleans the raw CSV, selects the five input variables and one
output variable required by the fuzzy system, and saves a
clean CSV ready for use in the Jupyter Notebook.

INPUT VARIABLES  : latitude, longitude, depth, gap, rms
OUTPUT VARIABLE  : mag  (earthquake magnitude)
"""

import os
import csv
import statistics

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
RAW_CSV    = os.path.join(BASE_DIR, "archive", "Japan earthquakes 2001 - 2018.csv")
CLEAN_CSV  = os.path.join(BASE_DIR, "earthquake_clean.csv")

# ── Columns we keep ───────────────────────────────────────────────────────────
INPUT_COLS  = ["latitude", "longitude", "depth", "gap", "rms"]
OUTPUT_COL  = "mag"
KEEP_COLS   = INPUT_COLS + [OUTPUT_COL]

# ── Valid ranges (hard physics / domain bounds) ────────────────────────────────
# Values outside these bounds are treated as sensor / recording errors.
VALID_RANGE = {
    "latitude"  : (23.0, 51.0),
    "longitude" : (124.0, 160.0),
    "depth"     : (0.0, 700.0),     # deepest subduction zones ~700 km
    "gap"       : (0.0, 360.0),     # azimuthal gap in degrees
    "rms"       : (0.0, 5.0),       # travel-time RMS residual (seconds)
    "mag"       : (4.5, 10.0),      # dataset already filtered ≥ 4.5
}


def try_float(value: str):
    """Return float or None if the string is empty / non-numeric."""
    v = value.strip()
    if v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def in_range(value: float, col: str) -> bool:
    lo, hi = VALID_RANGE[col]
    return lo <= value <= hi


def load_and_clean(raw_path: str):
    """
    Reads the raw CSV, keeps only the required columns, drops rows
    with missing or out-of-range values, and returns a list of dicts.
    """
    rows_raw        = 0
    rows_missing    = 0
    rows_out_range  = 0
    clean_rows      = []

    with open(raw_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows_raw += 1

            # ── 1. Parse all required columns ─────────────────────────────
            parsed = {}
            has_missing = False
            for col in KEEP_COLS:
                val = try_float(row.get(col, ""))
                if val is None:
                    has_missing = True
                    break
                parsed[col] = val

            if has_missing:
                rows_missing += 1
                continue

            # ── 2. Range validation ────────────────────────────────────────
            out_of_range = False
            for col in KEEP_COLS:
                if not in_range(parsed[col], col):
                    out_of_range = True
                    break

            if out_of_range:
                rows_out_range += 1
                continue

            clean_rows.append(parsed)

    return clean_rows, rows_raw, rows_missing, rows_out_range


def print_summary(clean_rows, rows_raw, rows_missing, rows_out_range):
    rows_clean = len(clean_rows)
    print("=" * 60)
    print("  DATA CLEANING SUMMARY")
    print("=" * 60)
    print(f"  Raw rows           : {rows_raw:>7,}")
    print(f"  Dropped (missing)  : {rows_missing:>7,}")
    print(f"  Dropped (out-range): {rows_out_range:>7,}")
    print(f"  Clean rows         : {rows_clean:>7,}")
    print()
    print("  Per-column statistics (clean data):")
    print(f"  {'Column':<12}  {'Min':>10}  {'Max':>10}  {'Mean':>10}  {'Stdev':>10}")
    print("  " + "-" * 58)
    for col in KEEP_COLS:
        vals  = [r[col] for r in clean_rows]
        mean  = statistics.mean(vals)
        stdev = statistics.stdev(vals)
        print(f"  {col:<12}  {min(vals):>10.4f}  {max(vals):>10.4f}  {mean:>10.4f}  {stdev:>10.4f}")
    print("=" * 60)


def save_clean_csv(clean_rows, out_path: str):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=KEEP_COLS)
        writer.writeheader()
        writer.writerows(clean_rows)
    print(f"\n  [OK] Clean CSV saved -> {out_path}")
    print(f"    ({len(clean_rows):,} rows × {len(KEEP_COLS)} columns)")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nLoading and cleaning dataset …\n")
    clean_rows, rows_raw, rows_missing, rows_out_range = load_and_clean(RAW_CSV)
    print_summary(clean_rows, rows_raw, rows_missing, rows_out_range)
    save_clean_csv(clean_rows, CLEAN_CSV)
    print("\nDone. The clean file is ready for Member 2's Jupyter Notebook.\n")
