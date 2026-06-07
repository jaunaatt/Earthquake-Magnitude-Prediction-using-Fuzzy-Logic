# Member 1 Deliverable: Fuzzy Logic System Design
## Earthquake Magnitude Prediction — Mathematical Foundation

**Course:** DKA Tugas Besar  
**Dataset:** Japan Earthquakes 2001–2018  
**Dataset Source:** [USGS Earthquake Hazards Program via Kaggle](https://www.kaggle.com/datasets/usgs/earthquake-database)  
**Author:** Member 1 — The Architect  
**Date:** June 2026  

---

## 1. Problem Definition

**Task:** Predict the **magnitude** (`mag`) of an earthquake using five measurable seismic attributes as inputs.  
**Type:** Fuzzy regression (continuous output).  
**Performance metric:** MAE (Mean Absolute Error) and MSE (Mean Squared Error), since `mag` is a continuous value with available ground truth labels (Member 3 task).

---

## 2. Dataset Summary

| Attribute | Value |
|---|---|
| Source file | `Japan earthquakes 2001 - 2018.csv` |
| Raw rows | 14,092 |
| Rows after cleaning | **13,233** |
| Rows dropped (missing `gap`/`rms`) | 859 |
| Input columns used | `latitude`, `longitude`, `depth`, `gap`, `rms` |
| Output column | `mag` |

> **Cleaning rules applied (see `data_preparation.py`):**
> 1. Drop any row where any of the 6 required columns is empty or non-numeric.
> 2. Drop rows where values fall outside hard physics bounds (e.g., depth > 700 km).
> 3. All remaining 13,233 rows are written to `earthquake_clean.csv`.

---

## 3. Variable Definitions & Data Ranges

The table below summarises the observed statistics from the **clean dataset**, used to anchor membership function boundaries.

| Variable | Unit | Min | p05 | p25 | Median | p75 | p95 | Max |
|---|---|---|---|---|---|---|---|---|
| `latitude`  | °N | 23.53 | 26.49 | 33.15 | 37.36 | 42.27 | 47.69 | 50.82 |
| `longitude` | °E | 124.35 | 129.53 | 141.07 | 142.45 | 144.43 | 154.84 | 158.82 |
| `depth`     | km | 0.00 | 10.00 | 14.40 | 35.00 | 50.37 | 153.50 | 683.36 |
| `gap`       | ° | 8.00 | 34.00 | 78.00 | 112.70 | 130.90 | 156.00 | 306.60 |
| `rms`       | s | 0.12 | 0.58 | 0.74 | 0.85 | 0.99 | 1.26 | 1.88 |
| `mag`       | Richter | 4.50 | 4.50 | 4.60 | 4.70 | 4.90 | 5.60 | 9.10 |

---

## 4. Linguistic Variables and Membership Functions

Each variable is partitioned into **three fuzzy sets** (Low / Medium / High) using **triangular** and **trapezoidal** membership functions. Triangular MFs are denoted `trimf(a, b, c)` and trapezoidal MFs `trapmf(a, b, c, d)`.

> **Convention:**  
> `trapmf(a, b, c, d)` → rises from 0 at `a` to 1 at `b`, stays 1 until `c`, falls to 0 at `d`.  
> `trimf(a, b, c)` → rises from 0 at `a` to 1 at `b`, falls to 0 at `c`.

---

### 4.1 Input Variable 1 — `latitude` (°N)

**Universe of discourse:** [23, 51]  
**Physical meaning:** Location north–south. Northern Japan (Hokkaido/Kuril arc) tends to have different seismicity than central/southern regions.

| Fuzzy Set | Label | MF Type | Parameters | Formula |
|---|---|---|---|---|
| LOW | South | trapmf | (23, 23, 30, 36) | flat at 1 below 30°N, taper to 0 at 36°N |
| MEDIUM | Central | trimf | (30, 37, 44) | peak at 37°N (mean of dataset) |
| HIGH | North | trapmf | (38, 44, 51, 51) | taper from 0 at 38°N, flat at 1 above 44°N |

**Mathematical definitions:**

```
μ_South(x)   = trapmf(x; 23, 23, 30, 36)
              = 1,                          if 23 ≤ x ≤ 30
              = (36 - x) / (36 - 30),      if 30 < x ≤ 36
              = 0,                          if x > 36

μ_Central(x) = trimf(x; 30, 37, 44)
              = (x - 30) / (37 - 30),      if 30 ≤ x ≤ 37
              = (44 - x) / (44 - 37),      if 37 < x ≤ 44
              = 0,                          otherwise

μ_North(x)   = trapmf(x; 38, 44, 51, 51)
              = 0,                          if x < 38
              = (x - 38) / (44 - 38),      if 38 ≤ x ≤ 44
              = 1,                          if x > 44
```

---

### 4.2 Input Variable 2 — `longitude` (°E)

**Universe of discourse:** [124, 159]  
**Physical meaning:** East–west location. The Pacific plate subduction zone dominates the eastern longitudes.

| Fuzzy Set | Label | MF Type | Parameters |
|---|---|---|---|
| LOW | West | trapmf | (124, 124, 131, 139) |
| MEDIUM | Central | trimf | (132, 142, 152) |
| HIGH | East | trapmf | (145, 152, 159, 159) |

**Mathematical definitions:**

```
μ_West(x)    = trapmf(x; 124, 124, 131, 139)
              = 1,                          if 124 ≤ x ≤ 131
              = (139 - x) / (139 - 131),   if 131 < x ≤ 139
              = 0,                          if x > 139

μ_Central(x) = trimf(x; 132, 142, 152)
              = (x - 132) / (142 - 132),   if 132 ≤ x ≤ 142
              = (152 - x) / (152 - 142),   if 142 < x ≤ 152
              = 0,                          otherwise

μ_East(x)    = trapmf(x; 145, 152, 159, 159)
              = 0,                          if x < 145
              = (x - 145) / (152 - 145),   if 145 ≤ x ≤ 152
              = 1,                          if x > 152
```

---

### 4.3 Input Variable 3 — `depth` (km)

**Universe of discourse:** [0, 700]  
**Physical meaning:** Shallow earthquakes (crustal, <70 km) tend to be more destructive; intermediate (70–300 km) and deep (>300 km) quakes travel through the mantle.

| Fuzzy Set | Label | MF Type | Parameters |
|---|---|---|---|
| LOW | Shallow | trapmf | (0, 0, 35, 70) |
| MEDIUM | Intermediate | trimf | (35, 150, 300) |
| HIGH | Deep | trapmf | (200, 350, 700, 700) |

**Mathematical definitions:**

```
μ_Shallow(x)      = trapmf(x; 0, 0, 35, 70)
                  = 1,                         if 0 ≤ x ≤ 35
                  = (70 - x) / (70 - 35),      if 35 < x ≤ 70
                  = 0,                          if x > 70

μ_Intermediate(x) = trimf(x; 35, 150, 300)
                  = (x - 35) / (150 - 35),     if 35 ≤ x ≤ 150
                  = (300 - x) / (300 - 150),   if 150 < x ≤ 300
                  = 0,                          otherwise

μ_Deep(x)         = trapmf(x; 200, 350, 700, 700)
                  = 0,                          if x < 200
                  = (x - 200) / (350 - 200),   if 200 ≤ x ≤ 350
                  = 1,                          if x > 350
```

---

### 4.4 Input Variable 4 — `gap` (°)

**Universe of discourse:** [0, 310]  
**Physical meaning:** Azimuthal gap — the largest angular gap in station coverage. A smaller gap → more complete network coverage → better magnitude estimation reliability.

| Fuzzy Set | Label | MF Type | Parameters |
|---|---|---|---|
| LOW | Well-covered | trapmf | (0, 0, 60, 90) |
| MEDIUM | Moderate | trimf | (60, 110, 160) |
| HIGH | Poorly-covered | trapmf | (130, 170, 310, 310) |

**Mathematical definitions:**

```
μ_WellCovered(x) = trapmf(x; 0, 0, 60, 90)
                 = 1,                        if 0 ≤ x ≤ 60
                 = (90 - x) / (90 - 60),    if 60 < x ≤ 90
                 = 0,                         if x > 90

μ_Moderate(x)    = trimf(x; 60, 110, 160)
                 = (x - 60) / (110 - 60),   if 60 ≤ x ≤ 110
                 = (160 - x) / (160 - 110), if 110 < x ≤ 160
                 = 0,                         otherwise

μ_PoorlyCovered(x) = trapmf(x; 130, 170, 310, 310)
                   = 0,                        if x < 130
                   = (x - 130) / (170 - 130), if 130 ≤ x ≤ 170
                   = 1,                         if x > 170
```

---

### 4.5 Input Variable 5 — `rms` (seconds)

**Universe of discourse:** [0.1, 1.9]  
**Physical meaning:** Root-mean-square travel-time residual. A low RMS indicates that seismic wave arrivals match the computed travel times well, implying a high-quality, well-located event.

| Fuzzy Set | Label | MF Type | Parameters |
|---|---|---|---|
| LOW | HighQuality | trapmf | (0.10, 0.10, 0.65, 0.80) |
| MEDIUM | Moderate | trimf | (0.65, 0.88, 1.10) |
| HIGH | LowQuality | trapmf | (0.95, 1.20, 1.90, 1.90) |

**Mathematical definitions:**

```
μ_HighQuality(x) = trapmf(x; 0.10, 0.10, 0.65, 0.80)
                 = 1,                              if 0.10 ≤ x ≤ 0.65
                 = (0.80 - x) / (0.80 - 0.65),   if 0.65 < x ≤ 0.80
                 = 0,                               if x > 0.80

μ_Moderate(x)    = trimf(x; 0.65, 0.88, 1.10)
                 = (x - 0.65) / (0.88 - 0.65),   if 0.65 ≤ x ≤ 0.88
                 = (1.10 - x) / (1.10 - 0.88),   if 0.88 < x ≤ 1.10
                 = 0,                               otherwise

μ_LowQuality(x)  = trapmf(x; 0.95, 1.20, 1.90, 1.90)
                 = 0,                               if x < 0.95
                 = (x - 0.95) / (1.20 - 0.95),   if 0.95 ≤ x ≤ 1.20
                 = 1,                               if x > 1.20
```

---

### 4.6 Output Variable — `mag` (Richter scale)

**Universe of discourse:** [4.5, 9.5]  
**Physical meaning:** Earthquake magnitude. Dataset contains events ≥ 4.5 (minimum reporting threshold).

| Fuzzy Set | Label | MF Type | Parameters |
|---|---|---|---|
| LOW | Minor | trapmf | (4.5, 4.5, 4.7, 5.0) |
| MEDIUM | Moderate | trimf | (4.7, 5.0, 5.8) |
| HIGH | Strong | trimf | (5.4, 6.0, 7.0) |
| VERY_HIGH | Major | trapmf | (6.5, 7.5, 9.5, 9.5) |

**Mathematical definitions:**

```
μ_Minor(x)    = trapmf(x; 4.5, 4.5, 4.7, 5.0)
              = 1,                           if 4.5 ≤ x ≤ 4.7
              = (5.0 - x) / (5.0 - 4.7),   if 4.7 < x ≤ 5.0
              = 0,                            if x > 5.0

μ_Moderate(x) = trimf(x; 4.7, 5.0, 5.8)
              = (x - 4.7) / (5.0 - 4.7),   if 4.7 ≤ x ≤ 5.0
              = (5.8 - x) / (5.8 - 5.0),   if 5.0 < x ≤ 5.8
              = 0,                            otherwise

μ_Strong(x)   = trimf(x; 5.4, 6.0, 7.0)
              = (x - 5.4) / (6.0 - 5.4),   if 5.4 ≤ x ≤ 6.0
              = (7.0 - x) / (7.0 - 6.0),   if 6.0 < x ≤ 7.0
              = 0,                            otherwise

μ_Major(x)    = trapmf(x; 6.5, 7.5, 9.5, 9.5)
              = 0,                            if x < 6.5
              = (x - 6.5) / (7.5 - 6.5),   if 6.5 ≤ x ≤ 7.5
              = 1,                            if x > 7.5
```

---

## 5. Rule Base (Minimum 15 Rules)

The following 20 fuzzy IF–THEN rules connect the five input variables to the output magnitude. Rules are written in the standard Mamdani / Sugeno format.

> **Notation shorthand:**  
> `lat`: latitude, `lon`: longitude, `dep`: depth, `gap`: gap, `rms`: rms, `mag`: output magnitude  
> `Lo/S` = Low/South/Shallow/Well-covered/HighQuality  
> `Me/C` = Medium/Central/Intermediate/Moderate  
> `Hi/N` = High/North/Deep/PoorlyCovered/LowQuality  

---

### Rule Set

| # | IF (latitude) | AND (longitude) | AND (depth) | AND (gap) | AND (rms) | THEN (mag) |
|---|---|---|---|---|---|---|
| R01 | South | West | Shallow | Well-covered | HighQuality | Minor |
| R02 | South | West | Shallow | Well-covered | LowQuality | Moderate |
| R03 | South | West | Intermediate | Moderate | Moderate | Moderate |
| R04 | South | East | Shallow | PoorlyCovered | LowQuality | Moderate |
| R05 | Central | Central | Shallow | Well-covered | HighQuality | Minor |
| R06 | Central | Central | Intermediate | Moderate | Moderate | Moderate |
| R07 | Central | East | Deep | PoorlyCovered | Moderate | Moderate |
| R08 | Central | Central | Shallow | PoorlyCovered | LowQuality | Strong |
| R09 | Central | East | Intermediate | Well-covered | HighQuality | Moderate |
| R10 | North | East | Shallow | Moderate | Moderate | Strong |
| R11 | North | East | Deep | Well-covered | HighQuality | Moderate |
| R12 | North | East | Deep | PoorlyCovered | LowQuality | Strong |
| R13 | Central | Central | Deep | Well-covered | HighQuality | Moderate |
| R14 | South | Central | Shallow | Well-covered | LowQuality | Minor |
| R15 | North | Central | Intermediate | Moderate | HighQuality | Strong |
| R16 | North | East | Shallow | PoorlyCovered | LowQuality | Major |
| R17 | Central | East | Shallow | Well-covered | HighQuality | Minor |
| R18 | South | East | Deep | Well-covered | Moderate | Moderate |
| R19 | Central | Central | Intermediate | PoorlyCovered | LowQuality | Strong |
| R20 | North | Central | Deep | PoorlyCovered | LowQuality | Major |

---

### Rule Justification / Rationale

| Rule(s) | Seismological Rationale |
|---|---|
| R01, R05, R14, R17 | Well-covered networks + high-quality signals + shallow-to-surface-close events near densely monitored areas → magnitudes are typically minor (recording threshold events). |
| R02, R04 | Low-quality RMS in the same shallow zone → harder-to-constrain magnitude, slight upward bias. |
| R06, R09, R13 | Intermediate-depth events along the subduction interface → moderate magnitudes dominate (most common class). |
| R08, R10, R15, R19 | Poor station coverage with either northern arc location or intermediate depth → elevated estimated magnitude since limited triangulation leads to stronger apparent shaking. |
| R16, R20 | Northern arc + poorly-covered + low-quality signal → potential major earthquake (Kuril/Kamchatka zone, site of many M7+ events in the dataset). |
| R03, R07, R18 | Mixed conditions (intermediate gap/rms) → moderate outcome, the default class given ~70% of events are M4.5–5.0. |
| R11, R12 | Deep northern events: even with poor coverage, deep quakes attenuate energy, keeping mag at Strong or below for most events. R12 bumps to Strong due to added noise from poor coverage. |

---

## 6. Sugeno Consequents (for Member 2's Sugeno Implementation)

In the Sugeno model, each rule's consequent is a **constant (zero-order Sugeno)**:

| Output Fuzzy Set | Crisp Constant (k) |
|---|---|
| Minor | **4.65** (midpoint of Minor MF core: 4.5–4.7) |
| Moderate | **5.00** (peak of Moderate MF) |
| Strong | **6.00** (peak of Strong MF) |
| Major | **7.80** (centroid of Major MF plateau) |

Map each rule's consequent label to these constants:

| Rule | Consequent Label | k value |
|---|---|---|
| R01 | Minor | 4.65 |
| R02 | Moderate | 5.00 |
| R03 | Moderate | 5.00 |
| R04 | Moderate | 5.00 |
| R05 | Minor | 4.65 |
| R06 | Moderate | 5.00 |
| R07 | Moderate | 5.00 |
| R08 | Strong | 6.00 |
| R09 | Moderate | 5.00 |
| R10 | Strong | 6.00 |
| R11 | Moderate | 5.00 |
| R12 | Strong | 6.00 |
| R13 | Moderate | 5.00 |
| R14 | Minor | 4.65 |
| R15 | Strong | 6.00 |
| R16 | Major | 7.80 |
| R17 | Minor | 4.65 |
| R18 | Moderate | 5.00 |
| R19 | Strong | 6.00 |
| R20 | Major | 7.80 |

---

## 7. Defuzzification Formulas (Reference for Member 2)

### 7.1 Mamdani — Center of Gravity (CoG)

After the inference engine clips (min) or scales (product) each output MF, the defuzzified output is:

$$
z^* = \frac{\int z \cdot \mu_{agg}(z) \, dz}{\int \mu_{agg}(z) \, dz}
$$

In practice, this integral is approximated by discretising `z` over `[4.5, 9.5]` with e.g. 1000 points and using the Riemann sum:

```
z_points = [4.5 + i*(9.5-4.5)/1000  for i in range(1001)]
numerator   = sum(z * mu_agg(z) for z in z_points)
denominator = sum(mu_agg(z)     for z in z_points)
z_star = numerator / denominator   (if denominator > 0 else 0)
```

### 7.2 Sugeno — Weighted Average

$$
z^* = \frac{\sum_{i=1}^{N} w_i \cdot k_i}{\sum_{i=1}^{N} w_i}
$$

where `w_i` is the **firing strength** of rule *i* (product or min of antecedent memberships) and `k_i` is the corresponding constant consequent from the table above.

---

## 8. Summary Checklist (Member 1 Completion)

- [x] **Dataset link documented** — USGS / Kaggle (see header)
- [x] **Dataset preparation** — `data_preparation.py` → `earthquake_clean.csv` (13,233 rows × 6 cols)
- [x] **Input variables defined** — `latitude`, `longitude`, `depth`, `gap`, `rms`
- [x] **Output variable defined** — `mag`
- [x] **Linguistic variables designed** — 3 sets per input, 4 sets for output
- [x] **Membership function parameters calculated** — triangular + trapezoidal, grounded in dataset percentiles
- [x] **Full mathematical MF formulas written** — Sections 4.1–4.6
- [x] **Rule base formulated** — 20 rules (≥ 15 required), with seismological rationale
- [x] **Sugeno constants provided** — Table in Section 6
- [x] **Defuzzification formulas provided** — CoG (Mamdani) + Weighted Average (Sugeno), Section 7
