# Motivational Salience Index (MSI) Modeling / preliminary analyses

This repository contains the machine learning preliminary analysis pipeline used to decode subjective food preference from implicit behavioral task signatures. The data are derived from Najberg et al. (2019), they contain: food item liking (VAS ratings), and behavioral responses to tasks featuring these food items. We compared the performance of reuglarized regression (Elastic Net), a tree-based model (HistGB), and recurrent neural networks (LSTM) in predicting food preferences. 

## Project Overview
The goal of this project is to predict individual food liking (VAS scores) using behavioral metrics derived from behavioral tasks. We applied within-person centering to isolate cue-specific reactivity.

## Key Results
- **Winning Model:** ElasticNet (Regularized Linear Regression)
- **Performance:** Mean within-person Spearman **ρ = 0.12** (Chance ρ = -0.02)
- **Primary Predictors:** Stimulus-specific deviations in False Alarm rates and Go-trial reaction times at the GNG task.

## Repository Structure
- `src/`: Python scripts for data extraction, preprocessing, and modeling.
- `data/`: (Local only) Raw and preprocessed datasets.
- `results/`: Performance plots and feature importance tables.
- `requirements.txt`: Python dependencies.

## How to Run
1. **Setup:** `pip install -r requirements.txt`
2. **Preprocess:** Run `01_data_extraction.py` through `03_by_item_imputation_normalization.py`.
3. **Model:** Run `04_ML_modelfitting.py` to replicate the ElasticNet results.
