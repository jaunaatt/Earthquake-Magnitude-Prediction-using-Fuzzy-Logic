# Project Planning: Fuzzy Logic Implementation for Earthquake Magnitude Prediction

**Course:** DKA (Tugas Besar)  
**Dataset:** Japan Earthquakes 2001 - 2018 (14,092 rows)  
**Team Size:** 3 Members  
**Submission Deadline:** June 15, 2026  

---

## 🧑‍💻 Member 1: The Architect (Data & Logic Design)
**Focus:** Project setup, data handling, and designing the mathematical foundation.

### Tasks:
- [x] **Administration:** Get topic approved and register the group and dataset link in `LIST OF TUBES DKA 2526.xlsx`.
- [x] **Dataset Preparation:** Clean the CSV file and define input variables (`latitude`, `longitude`, `depth`, `gap`, `rms`) and target output (`mag`). → See `data_preparation.py` & `earthquake_clean.csv` (13,233 rows).
- [x] **Fuzzy Logic System Design (Core Task):** - Define the linguistic variables for all inputs and outputs.
    - Design and calculate the mathematical ranges for the Membership Functions.
    - Formulate the Rule Base with a minimum of 15 logical rules connecting inputs to the output magnitude.
    → See `fuzzy_design.md` (20 rules, full MF math, Sugeno constants, defuzz formulas).

---

## 👨‍💻 Member 2: The Lead Developer (Python Implementation)
**Focus:** Translating the mathematical design into pure, from-scratch Python code.

### Tasks:
- [x] **Environment Setup:** Create the `.ipynb` Jupyter Notebook file.
- [x] **Fuzzification:** Write python functions from scratch to translate crisp CSV input values into fuzzy membership degrees.
- [x] **Inference Engine:** Write the logic to apply the 15+ rules for both **Mamdani** and **Sugeno** methods.
- [x] **Defuzzification:** Code the final output formulas (e.g., Center of Gravity for Mamdani, Weighted Average for Sugeno).
- [x] *Rule Check:* Ensure absolute ZERO usage of external fuzzy logic libraries.

---

## 🕵️‍♂️ Member 3: The Analyst (Evaluation & Final Deliverables)
**Focus:** Testing the system, calculating errors, and compiling the final submission.

### Tasks:
- [x] **Performance Evaluation:** Write code to calculate Error Metrics (MSE or MAE) comparing the fuzzy system's predicted magnitude against the actual magnitude in the CSV.
- [x] **Analysis:** Document the differences in output results and write an interpretation of the advantages and disadvantages of Mamdani vs. Sugeno based on the metrics.
- [x] **Final Deliverables:** Compile the design, analysis, and execution instructions into the final `.pdf` report. (Code and Notebook completed).
- [ ] **Bonus (Optional):** Take the lead on developing the Streamlit web app UI (+5 points) or integrating Machine/Deep Learning (+10 to +20 points). (Skipped).

---

## ⚠️ Crucial Team Milestones & Reminders
- [ ] **Knowledge Transfer Meeting:** Schedule a 1-hour sync a few days before June 15th so each member can present their part.
- [ ] **Week 15 Q&A Session:** ALL members must be present (on-site or offline) for the 15-minute live review with the lecturer. No slides required, but everyone must understand the code and logic.
