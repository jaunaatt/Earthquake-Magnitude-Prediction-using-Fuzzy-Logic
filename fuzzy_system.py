"""
=============================================================
  Member 2 – Fuzzy Logic Implementation (from scratch)
  Project : Earthquake Magnitude Prediction
  Dataset : Japan Earthquakes 2001-2018  (earthquake_clean.csv)
  Design  : fuzzy_design.md

  Methods implemented
  -------------------
    1. Fuzzification   – crisp CSV values  →  fuzzy membership degrees
    2. Mamdani Inference + Center-of-Gravity defuzzification
    3. Sugeno  Inference + Weighted-Average  defuzzification

  IMPORTANT: Zero external fuzzy-logic libraries are used.
             Only Python built-ins + csv / math.
=============================================================
"""

import csv
import math
import os

# ─────────────────────────────────────────────────────────────────────────────
# 0.  FILE PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CLEAN_CSV  = os.path.join(BASE_DIR, "earthquake_clean.csv")


# =============================================================================
# 1.  MEMBERSHIP FUNCTION PRIMITIVES
# =============================================================================

def trimf(x: float, a: float, b: float, c: float) -> float:
    """
    Triangular membership function.

    Shape:
        0     for x <= a
        (x-a)/(b-a)   for a < x <= b
        (c-x)/(c-b)   for b < x <= c
        0     for x > c

    Parameters
    ----------
    x : crisp input value
    a : left foot  (membership = 0)
    b : peak       (membership = 1)
    c : right foot (membership = 0)
    """
    if b == a and b == c:          # degenerate singleton
        return 1.0 if x == b else 0.0
    if x <= a or x >= c:
        return 0.0
    if x <= b:
        denom = b - a
        return (x - a) / denom if denom != 0 else 1.0
    else:
        denom = c - b
        return (c - x) / denom if denom != 0 else 1.0


def trapmf(x: float, a: float, b: float, c: float, d: float) -> float:
    """
    Trapezoidal membership function.

    Shape:
        0           for x <= a
        (x-a)/(b-a) for a < x <= b    (rising slope)
        1           for b < x <= c    (flat top)
        (d-x)/(d-c) for c < x <= d    (falling slope)
        0           for x > d

    Parameters
    ----------
    x : crisp input value
    a : left foot  (membership starts rising)
    b : left shoulder (membership = 1 from here …)
    c : right shoulder (… to here)
    d : right foot (membership = 0)
    """
    if x <= a or x > d:
        return 0.0
    if x <= b:
        denom = b - a
        return (x - a) / denom if denom != 0 else 1.0
    if x <= c:
        return 1.0
    # c < x <= d
    denom = d - c
    return (d - x) / denom if denom != 0 else 0.0


# =============================================================================
# 2.  FUZZIFICATION
#     Translate one crisp CSV row into a dict of membership degrees.
# =============================================================================

# ── 2.1  latitude  [23, 51] ──────────────────────────────────────────────────
def fuzzify_latitude(lat: float) -> dict:
    """
    Returns:
        {'South':   μ,   # trapmf(23, 23, 30, 36)
         'Central': μ,   # trimf (30, 37, 44)
         'North':   μ}   # trapmf(38, 44, 51, 51)
    """
    return {
        'South':   trapmf(lat, 23, 23, 30, 36),
        'Central': trimf (lat, 30, 37, 44),
        'North':   trapmf(lat, 38, 44, 51, 51),
    }


# ── 2.2  longitude  [124, 159] ───────────────────────────────────────────────
def fuzzify_longitude(lon: float) -> dict:
    """
    Returns:
        {'West':    μ,   # trapmf(124, 124, 131, 139)
         'Central': μ,   # trimf (132, 142, 152)
         'East':    μ}   # trapmf(145, 152, 159, 159)
    """
    return {
        'West':    trapmf(lon, 124, 124, 131, 139),
        'Central': trimf (lon, 132, 142, 152),
        'East':    trapmf(lon, 145, 152, 159, 159),
    }


# ── 2.3  depth  [0, 700] km ──────────────────────────────────────────────────
def fuzzify_depth(dep: float) -> dict:
    """
    Returns:
        {'Shallow':      μ,   # trapmf(0,   0,   35,  70)
         'Intermediate': μ,   # trimf (35,  150, 300)
         'Deep':         μ}   # trapmf(200, 350, 700, 700)
    """
    return {
        'Shallow':      trapmf(dep,   0,   0,   35,  70),
        'Intermediate': trimf (dep,  35, 150, 300),
        'Deep':         trapmf(dep, 200, 350, 700, 700),
    }


# ── 2.4  gap  [0, 310] degrees ───────────────────────────────────────────────
def fuzzify_gap(gap: float) -> dict:
    """
    Returns:
        {'WellCovered':   μ,   # trapmf(0,   0,   60,  90)
         'Moderate':      μ,   # trimf (60,  110, 160)
         'PoorlyCovered': μ}   # trapmf(130, 170, 310, 310)
    """
    return {
        'WellCovered':   trapmf(gap,   0,   0,  60,  90),
        'Moderate':      trimf (gap,  60, 110, 160),
        'PoorlyCovered': trapmf(gap, 130, 170, 310, 310),
    }


# ── 2.5  rms  [0.1, 1.9] seconds ─────────────────────────────────────────────
def fuzzify_rms(rms: float) -> dict:
    """
    Returns:
        {'HighQuality': μ,   # trapmf(0.10, 0.10, 0.65, 0.80)
         'Moderate':    μ,   # trimf (0.65, 0.88, 1.10)
         'LowQuality':  μ}   # trapmf(0.95, 1.20, 1.90, 1.90)
    """
    return {
        'HighQuality': trapmf(rms, 0.10, 0.10, 0.65, 0.80),
        'Moderate':    trimf (rms, 0.65, 0.88, 1.10),
        'LowQuality':  trapmf(rms, 0.95, 1.20, 1.90, 1.90),
    }


def fuzzify_row(row: dict) -> dict:
    """
    Master fuzzification function.
    Takes a dict with keys: latitude, longitude, depth, gap, rms.
    Returns a nested dict:
        {
          'latitude':  {'South': μ, 'Central': μ, 'North': μ},
          'longitude': {'West': μ, 'Central': μ, 'East': μ},
          'depth':     {'Shallow': μ, 'Intermediate': μ, 'Deep': μ},
          'gap':       {'WellCovered': μ, 'Moderate': μ, 'PoorlyCovered': μ},
          'rms':       {'HighQuality': μ, 'Moderate': μ, 'LowQuality': μ},
        }
    """
    return {
        'latitude':  fuzzify_latitude (float(row['latitude'])),
        'longitude': fuzzify_longitude(float(row['longitude'])),
        'depth':     fuzzify_depth    (float(row['depth'])),
        'gap':       fuzzify_gap      (float(row['gap'])),
        'rms':       fuzzify_rms      (float(row['rms'])),
    }


# =============================================================================
# 3.  RULE BASE  (20 rules, identical for both Mamdani and Sugeno)
#
#     Each rule is a tuple:
#       (lat_set, lon_set, dep_set, gap_set, rms_set, output_set)
#
#     Antecedent firing strength = min(μ_lat, μ_lon, μ_dep, μ_gap, μ_rms)
# =============================================================================

RULES = [
    # (lat,      lon,       dep,            gap,              rms,           output)
    ('South',   'West',    'Shallow',      'WellCovered',   'HighQuality', 'Minor'),     # R01
    ('South',   'West',    'Shallow',      'WellCovered',   'LowQuality',  'Moderate'),  # R02
    ('South',   'West',    'Intermediate', 'Moderate',      'Moderate',    'Moderate'),  # R03
    ('South',   'East',    'Shallow',      'PoorlyCovered', 'LowQuality',  'Moderate'),  # R04
    ('Central', 'Central', 'Shallow',      'WellCovered',   'HighQuality', 'Minor'),     # R05
    ('Central', 'Central', 'Intermediate', 'Moderate',      'Moderate',    'Moderate'),  # R06
    ('Central', 'East',    'Deep',         'PoorlyCovered', 'Moderate',    'Moderate'),  # R07
    ('Central', 'Central', 'Shallow',      'PoorlyCovered', 'LowQuality',  'Strong'),    # R08
    ('Central', 'East',    'Intermediate', 'WellCovered',   'HighQuality', 'Moderate'),  # R09
    ('North',   'East',    'Shallow',      'Moderate',      'Moderate',    'Strong'),    # R10
    ('North',   'East',    'Deep',         'WellCovered',   'HighQuality', 'Moderate'),  # R11
    ('North',   'East',    'Deep',         'PoorlyCovered', 'LowQuality',  'Strong'),    # R12
    ('Central', 'Central', 'Deep',         'WellCovered',   'HighQuality', 'Moderate'),  # R13
    ('South',   'Central', 'Shallow',      'WellCovered',   'LowQuality',  'Minor'),     # R14
    ('North',   'Central', 'Intermediate', 'Moderate',      'HighQuality', 'Strong'),    # R15
    ('North',   'East',    'Shallow',      'PoorlyCovered', 'LowQuality',  'Major'),     # R16
    ('Central', 'East',    'Shallow',      'WellCovered',   'HighQuality', 'Minor'),     # R17
    ('South',   'East',    'Deep',         'WellCovered',   'Moderate',    'Moderate'),  # R18
    ('Central', 'Central', 'Intermediate', 'PoorlyCovered', 'LowQuality',  'Strong'),   # R19
    ('North',   'Central', 'Deep',         'PoorlyCovered', 'LowQuality',  'Major'),    # R20
    # ── Additional rules to fill common coverage gaps ──────────────────────────
    ('Central', 'Central', 'Shallow',      'Moderate',      'Moderate',    'Minor'),    # R21
    ('Central', 'Central', 'Shallow',      'Moderate',      'HighQuality', 'Minor'),    # R22
    ('Central', 'Central', 'Shallow',      'WellCovered',   'Moderate',    'Minor'),    # R23
    ('South',   'Central', 'Shallow',      'Moderate',      'Moderate',    'Minor'),    # R24
    ('North',   'Central', 'Shallow',      'Moderate',      'Moderate',    'Moderate'), # R25
]


def _firing_strength(fuzzy_inputs: dict, rule: tuple) -> float:
    """
    Compute the firing strength (w) of a single rule using the MIN operator.

    Parameters
    ----------
    fuzzy_inputs : output of fuzzify_row()
    rule         : one element of RULES

    Returns
    -------
    w : float in [0, 1]
    """
    lat_set, lon_set, dep_set, gap_set, rms_set, _ = rule
    mu_lat = fuzzy_inputs['latitude' ][lat_set]
    mu_lon = fuzzy_inputs['longitude'][lon_set]
    mu_dep = fuzzy_inputs['depth'    ][dep_set]
    mu_gap = fuzzy_inputs['gap'      ][gap_set]
    mu_rms = fuzzy_inputs['rms'      ][rms_set]
    return min(mu_lat, mu_lon, mu_dep, mu_gap, mu_rms)


# =============================================================================
# 4.  OUTPUT MEMBERSHIP FUNCTIONS  (for Mamdani defuzzification)
# =============================================================================

def mu_output(z: float, label: str) -> float:
    """
    Evaluate the output membership function for a given label at point z.

    Output universe of discourse: [4.5, 9.5]

    Labels and MF parameters (from design document §4.6):
      Minor    : trapmf(4.5, 4.5, 4.7, 5.0)
      Moderate : trimf (4.7, 5.0, 5.8)
      Strong   : trimf (5.4, 6.0, 7.0)
      Major    : trapmf(6.5, 7.5, 9.5, 9.5)
    """
    if label == 'Minor':
        return trapmf(z, 4.5, 4.5, 4.7, 5.0)
    elif label == 'Moderate':
        return trimf(z, 4.7, 5.0, 5.8)
    elif label == 'Strong':
        return trimf(z, 5.4, 6.0, 7.0)
    elif label == 'Major':
        return trapmf(z, 6.5, 7.5, 9.5, 9.5)
    else:
        raise ValueError(f"Unknown output label: '{label}'")


# Sugeno zero-order constants (design document §6)
SUGENO_CONSTANTS = {
    'Minor':    4.65,
    'Moderate': 5.00,
    'Strong':   6.00,
    'Major':    7.80,
}


# =============================================================================
# 5.  INFERENCE ENGINE
# =============================================================================

# ── 5.1  Mamdani Inference ───────────────────────────────────────────────────

def mamdani_inference(fuzzy_inputs: dict) -> dict:
    """
    Apply all 20 rules using the Mamdani (min) implication.

    For each rule i:
        w_i  = firing strength  (min of antecedent memberships)
        The consequent output MF is *clipped* at w_i.

    Returns
    -------
    activated_rules : list of (w, output_label) pairs for all rules with w > 0
    """
    activated = []
    for rule in RULES:
        w = _firing_strength(fuzzy_inputs, rule)
        if w > 0.0:
            output_label = rule[-1]
            activated.append((w, output_label))
    return activated


def mamdani_aggregate(activated_rules: list, z_points: list) -> list:
    """
    Aggregate all activated rule outputs into a single fuzzy set
    using the MAX aggregation operator.

    For each point z in z_points:
        μ_agg(z) = max over all active rules of  min(w_i, μ_output_i(z))

    Parameters
    ----------
    activated_rules : output of mamdani_inference()
    z_points        : discretised output universe

    Returns
    -------
    agg_values : list of μ_agg(z) for each z in z_points
    """
    agg_values = []
    for z in z_points:
        mu_agg = 0.0
        for (w, label) in activated_rules:
            clipped = min(w, mu_output(z, label))   # clip (Mamdani implication)
            if clipped > mu_agg:
                mu_agg = clipped                     # max aggregation
        agg_values.append(mu_agg)
    return agg_values


# ── 5.2  Sugeno Inference ────────────────────────────────────────────────────

def sugeno_inference(fuzzy_inputs: dict) -> list:
    """
    Apply all 20 rules using zero-order Sugeno.

    For each rule i:
        w_i = firing strength (min of antecedent memberships)
        k_i = crisp constant consequent (from SUGENO_CONSTANTS)

    Returns
    -------
    list of (w_i, k_i) for all rules with w_i > 0
    """
    weighted_outputs = []
    for rule in RULES:
        w = _firing_strength(fuzzy_inputs, rule)
        if w > 0.0:
            k = SUGENO_CONSTANTS[rule[-1]]
            weighted_outputs.append((w, k))
    return weighted_outputs


# =============================================================================
# 6.  DEFUZZIFICATION
# =============================================================================

# ── 6.1  Mamdani – Center of Gravity (CoG) ───────────────────────────────────

def defuzzify_mamdani_cog(activated_rules: list,
                           z_min: float = 4.5,
                           z_max: float = 9.5,
                           n_points: int = 1000) -> float:
    """
    Center of Gravity (centroid) defuzzification for Mamdani.

    Approximates the continuous integral using a Riemann sum over n_points
    equally-spaced points on [z_min, z_max]:

        z* = Σ z · μ_agg(z) Δz  /  Σ μ_agg(z) Δz

    Since Δz cancels, this simplifies to:

        z* = Σ z · μ_agg(z)  /  Σ μ_agg(z)

    Parameters
    ----------
    activated_rules : output of mamdani_inference()
    z_min, z_max    : output universe bounds
    n_points        : number of discretisation points (higher = more accurate)

    Returns
    -------
    z_star : defuzzified crisp magnitude, or (z_min+z_max)/2 if denominator=0
    """
    step     = (z_max - z_min) / n_points
    z_points = [z_min + i * step for i in range(n_points + 1)]

    agg_values  = mamdani_aggregate(activated_rules, z_points)

    numerator   = sum(z * mu for z, mu in zip(z_points, agg_values))
    denominator = sum(agg_values)

    if denominator == 0.0:
        # Fallback: no rules fired → return weighted centroid of output MF peaks
        # (Minor=4.65, Moderate=5.0, Strong=6.0, Major=7.8) weighted equally
        return (4.65 + 5.0 + 6.0 + 7.8) / 4.0
    return numerator / denominator


# ── 6.2  Sugeno – Weighted Average ───────────────────────────────────────────

def defuzzify_sugeno_weighted_avg(weighted_outputs: list) -> float:
    """
    Weighted average defuzzification for zero-order Sugeno.

        z* = Σ (w_i · k_i)  /  Σ w_i

    Parameters
    ----------
    weighted_outputs : list of (w_i, k_i) from sugeno_inference()

    Returns
    -------
    z_star : defuzzified crisp magnitude, or 0.0 if all weights are 0
    """
    if not weighted_outputs:
        # Fallback: no rules fired → return weighted centroid of all consequents
        return (4.65 + 5.0 + 6.0 + 7.8) / 4.0
    numerator   = sum(w * k for w, k in weighted_outputs)
    denominator = sum(w     for w, _ in weighted_outputs)
    if denominator == 0.0:
        return (4.65 + 5.0 + 6.0 + 7.8) / 4.0
    return numerator / denominator


# =============================================================================
# 7.  TOP-LEVEL PREDICTION FUNCTION
#     Runs both Mamdani and Sugeno end-to-end for a single input row.
# =============================================================================

def predict(row: dict) -> dict:
    """
    Full fuzzy pipeline for one earthquake record.

    Parameters
    ----------
    row : dict with keys 'latitude', 'longitude', 'depth', 'gap', 'rms'
          (and optionally 'mag' for ground-truth reference)

    Returns
    -------
    result : dict with keys
        'fuzzy_inputs'      – fuzzified membership degrees
        'mamdani_pred'      – defuzzified Mamdani magnitude
        'sugeno_pred'       – defuzzified Sugeno magnitude
        'actual_mag'        – ground-truth (float or None)
    """
    # Step 1 – Fuzzification
    fuzzy_inputs = fuzzify_row(row)

    # Step 2 – Mamdani Inference + Defuzzification
    mamdani_activated = mamdani_inference(fuzzy_inputs)
    mamdani_pred      = defuzzify_mamdani_cog(mamdani_activated)

    # Step 3 – Sugeno Inference + Defuzzification
    sugeno_outputs = sugeno_inference(fuzzy_inputs)
    sugeno_pred    = defuzzify_sugeno_weighted_avg(sugeno_outputs)

    # Ground truth (if available)
    actual = None
    if 'mag' in row and row['mag'] not in ('', None):
        try:
            actual = float(row['mag'])
        except ValueError:
            pass

    return {
        'fuzzy_inputs': fuzzy_inputs,
        'mamdani_pred': mamdani_pred,
        'sugeno_pred':  sugeno_pred,
        'actual_mag':   actual,
    }


# =============================================================================
# 8.  BATCH EVALUATION  (reads earthquake_clean.csv, runs both methods)
# =============================================================================

def evaluate_all(csv_path: str = CLEAN_CSV,
                 max_rows: int = None,
                 verbose: bool = False) -> dict:
    """
    Run both fuzzy systems over the full clean dataset and compute:
        - MAE  (Mean Absolute Error)
        - MSE  (Mean Squared Error)
        - RMSE (Root Mean Squared Error)

    Parameters
    ----------
    csv_path : path to earthquake_clean.csv
    max_rows : if set, only process the first N rows (for quick testing)
    verbose  : if True, print predictions row by row

    Returns
    -------
    metrics : dict with 'mamdani' and 'sugeno' sub-dicts containing
              MAE, MSE, RMSE, and n (number of evaluated rows)
    """
    mamdani_errors  = []
    sugeno_errors   = []

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break

            result = predict(row)
            actual = result['actual_mag']
            if actual is None:
                continue

            m_pred = result['mamdani_pred']
            s_pred = result['sugeno_pred']

            mamdani_errors.append(actual - m_pred)
            sugeno_errors.append(actual  - s_pred)

            if verbose:
                print(f"  Row {i+1:5d} | actual={actual:.2f} | "
                      f"mamdani={m_pred:.3f} | sugeno={s_pred:.3f}")

    def _metrics(errors):
        n   = len(errors)
        if n == 0:
            return {'n': 0, 'MAE': None, 'MSE': None, 'RMSE': None}
        mae  = sum(abs(e) for e in errors) / n
        mse  = sum(e**2   for e in errors) / n
        rmse = math.sqrt(mse)
        return {'n': n, 'MAE': mae, 'MSE': mse, 'RMSE': rmse}

    metrics = {
        'mamdani': _metrics(mamdani_errors),
        'sugeno':  _metrics(sugeno_errors),
    }
    return metrics


# =============================================================================
# 9.  DEMO / MAIN
# =============================================================================

def _demo_single_row():
    """Demonstrate the full pipeline on one hand-crafted example row."""
    print("=" * 65)
    print("  SINGLE-ROW DEMO")
    print("  (Central Japan, shallow crustal, well-monitored, high quality)")
    print("=" * 65)

    example = {
        'latitude':  37.36,    # Central
        'longitude': 142.45,   # Central
        'depth':     35.0,     # Shallow (median)
        'gap':       112.7,    # Moderate coverage (median)
        'rms':       0.85,     # Moderate quality (median)
        'mag':       4.7,      # ground truth
    }

    result = predict(example)

    print(f"\n  Input:")
    for k, v in example.items():
        if k != 'mag':
            print(f"    {k:<12}: {v}")
    print(f"    {'actual mag':<12}: {example['mag']}")

    print(f"\n  Fuzzy Membership Degrees:")
    for var, memberships in result['fuzzy_inputs'].items():
        sets_str = ',  '.join(f"{s}={mu:.4f}" for s, mu in memberships.items())
        print(f"    {var:<12}: {sets_str}")

    print(f"\n  Predictions:")
    print(f"    Mamdani  (CoG)           : {result['mamdani_pred']:.4f}")
    print(f"    Sugeno   (Weighted Avg)  : {result['sugeno_pred']:.4f}")
    print(f"    Actual magnitude         : {result['actual_mag']:.4f}")
    print(f"    Mamdani error            : {result['actual_mag'] - result['mamdani_pred']:+.4f}")
    print(f"    Sugeno  error            : {result['actual_mag'] - result['sugeno_pred']:+.4f}")
    print()


def _demo_batch(n=200):
    """Quick batch evaluation on the first n rows of the clean CSV."""
    print("=" * 65)
    print(f"  BATCH EVALUATION  (first {n} rows of earthquake_clean.csv)")
    print("=" * 65)

    metrics = evaluate_all(max_rows=n, verbose=False)

    for method in ('mamdani', 'sugeno'):
        m = metrics[method]
        label = "Mamdani (CoG)" if method == 'mamdani' else "Sugeno  (WA) "
        print(f"\n  {label}:")
        print(f"    Rows evaluated : {m['n']:,}")
        if m['MAE'] is not None:
            print(f"    MAE            : {m['MAE']:.4f}")
            print(f"    MSE            : {m['MSE']:.4f}")
            print(f"    RMSE           : {m['RMSE']:.4f}")
    print()


if __name__ == '__main__':
    _demo_single_row()
    _demo_batch(n=200)

    # ── Full dataset evaluation (may take a few minutes) ──────────────────
    print("=" * 65)
    print("  FULL DATASET EVALUATION  (13,233 rows)")
    print("=" * 65)
    full_metrics = evaluate_all(verbose=False)
    for method in ('mamdani', 'sugeno'):
        m = full_metrics[method]
        label = "Mamdani (CoG)" if method == 'mamdani' else "Sugeno  (WA) "
        print(f"\n  {label}:")
        print(f"    Rows evaluated : {m['n']:,}")
        if m['MAE'] is not None:
            print(f"    MAE            : {m['MAE']:.6f}")
            print(f"    MSE            : {m['MSE']:.6f}")
            print(f"    RMSE           : {m['RMSE']:.6f}")
    print()
    print("  Done. Import this module from your Jupyter Notebook with:")
    print("    from fuzzy_system import predict, evaluate_all")
