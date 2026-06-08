# Earthquake Magnitude Prediction using Fuzzy Logic

This repository contains our final project (Tugas Besar) for the DKA (Intro to AI) course. The goal of this project is to predict the magnitude of earthquakes based on seismic attributes using a custom-built Fuzzy Logic Inference System. 

**Note**: As per project constraints, the fuzzy logic system was implemented entirely **from scratch** using pure Python, without relying on external fuzzy logic libraries like `scikit-fuzzy`.

## 📊 Dataset
**Japan Earthquakes 2001 - 2018**
- **Source**: [Japan's Earthquakes 2001-2018 via Kaggle]([https://www.kaggle.com/datasets/usgs/earthquake-database](https://www.kaggle.com/datasets/aerodinamicc/earthquakes-in-japan))
- **Original Size**: 14,092 rows
- **Cleaned Size**: 13,233 rows
- **Input Features**: `latitude`, `longitude`, `depth`, `gap`, `rms`
- **Target Output**: `mag` (Magnitude)

## 🏗️ Project Structure
- `data_preparation.py`: Script to clean the raw dataset and enforce physics bounds. Outputs `earthquake_clean.csv`.
- `fuzzy_design.md`: Detailed documentation on the mathematical foundation of the fuzzy logic system, including the definition of linguistic variables, membership function equations, and the 20 rule bases.
- `fuzzy_system.py`: The core inference engine built from scratch. Contains the fuzzification, inference logic (Mamdani and Sugeno methods), and defuzzification algorithms.
- `fuzzy_earthquake_notebook.ipynb`: A Jupyter Notebook to run, test, and visualize the output of the fuzzy inference system.
- `Final_Analysis_Report.md`: The final performance evaluation report, comparing the Mean Absolute Error (MAE) and Mean Squared Error (MSE) of the Mamdani vs. Sugeno methods.
- `Project_Planning_DKA_Tubes.md`: The original project planning document tracking tasks and responsibilities.

## 🚀 How to Run
1. **Prerequisites**: Ensure you have Python and Jupyter Notebook installed. You will also need `pandas`, `numpy`, and `matplotlib` for data handling and visualization.
2. **Data Preparation**: Run `data_preparation.py` to generate the cleaned dataset `earthquake_clean.csv`.
3. **Fuzzy System**: Open `fuzzy_earthquake_notebook.ipynb` and run all cells. This notebook imports the custom logic from `fuzzy_system.py` and processes the dataset to produce magnitude predictions, along with visualizations.

## 👥 Team Responsibilities
- **Muhammad Athallah Alimusa**: Project setup, data cleaning, and mathematical fuzzy logic design.
- **Jauna At-Tijani Mamoko**: Translation of the mathematical design into pure Python code (Fuzzification, Inference Engine, Defuzzification).
- **Rakha Anargya Wibowo**: Performance evaluation, error metric calculation (MSE/MAE), comparative analysis of Mamdani vs Sugeno, and final report compilation.

## 📈 Fuzzy Analysis & Results

The fuzzy logic system was implemented using both **Mamdani** and **Sugeno** inference methods, allowing for a rigorous comparative analysis based on performance metrics (MAE and MSE) against the actual dataset.

### Performance Metrics
The system was evaluated on a subset of the data:
| Inference Method | Mean Absolute Error (MAE) | Mean Squared Error (MSE) |
| :--- | :---: | :---: |
| **Mamdani (CoG)** | **0.7522** | **1.0024** |
| **Sugeno (Weighted Avg)** | **0.6739** | **0.8427** |

![Prediction vs Actual](./pred_vs_actual.png)

### Comparative Analysis
- **Mamdani Method**: Highly intuitive and human-interpretable as the output fuzzy sets provide a full distribution of the prediction (see example below). However, it is computationally expensive due to the discrete area integration required for Center of Gravity.
- **Sugeno Method**: Significantly more computationally efficient and precise for quantitative data, evidenced by the lower error metrics. It replaces area integration with a simple weighted average, making it exponentially faster but slightly less intuitive.

![Mamdani Output Example](./mamdani_output.png)

### Membership Functions
The system utilizes 5 inputs, translated into linguistic variables using standard membership functions:
![Membership Functions](./mf_plots.png)

---
*Created for the Intro to AI (DKA) Course - June 2026*
