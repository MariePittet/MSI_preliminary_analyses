# -*- coding: utf-8 -*-
"""
Script: 04_ML_Modeling.py
Project: Motivational Salience Index (MSI)
Author: Marie Pittet
Description: Benchmarks ElasticNet, Ridge, and HistGB on item-level behavioral data.
"""
# ------------------------------------------------------------
# 0) Env
# ------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from scipy.stats import spearmanr

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet

# ------------------------------------------------------------
# 1) CONFIG & TRANSFORMERS
# ------------------------------------------------------------
DATA_PATH = "../data/preprocessed/by_item/training.csv"
TARGET, PERSON_ID, ITEM_ID = "vas_score", "fk_device_id", "item_id"
ID_COLS = [PERSON_ID, ITEM_ID]
DROP_SUBSTRINGS = ["SST", "n_trials"]
N_SPLITS, RT_QUANTILE, RANDOM_STATE = 5, 0.99, 42

# --- Human-Readable Mapping for Grant-Ready Figures ---
RENAME_DICT = {
    'n_fa_GNG_relative': 'Inhibitory Failure (GNG False Alarms)',
    'mean_rt_go_GNG_relative': 'Approach Speed (GNG Go RT)',
    'n_hit_GNG_relative': 'Task Engagement (GNG Hit Rate)',
    'n_hit_GNG': 'Total Successful Trials (GNG Hits)',
    'fa_rate_CAT_relative': 'Choice Impulsivity (CAT Errors)',
    'acc_nogo_CAT_relative': 'Self-Control Accuracy (CAT)',
    'median_rt_fa_CAT': 'Fast Decision Speed (CAT)',
    'median_rt_go_GNG_relative': 'Consistent Approach Speed (GNG)',
    'n_cr_CAT_relative': 'Successful Inhibitions (CAT)',
    'n_hit_CAT_relative': 'Selection Accuracy (CAT)',
    'n_miss_GNG_relative': 'Attention Lapses (GNG Misses)',
    'missingindicator_median_rt_fa_GNG_relative': 'Data Consistency Index',
}

class BehavioralCleaner(BaseEstimator, TransformerMixin):
    """Hard-removes all non-behavioral IDs and substrings to prevent leakage."""
    def __init__(self, substrings, ids_to_kill):
        self.substrings = substrings
        self.ids_to_kill = ids_to_kill

    def fit(self, X, y=None):
        cols = list(X.columns)
        self.cols_to_drop_ = [
            c for c in cols if any(s in c for s in self.substrings) 
            or any(i in c for i in self.ids_to_kill)
        ]
        return self

    def transform(self, X):
        return X.drop(columns=self.cols_to_drop_, errors="ignore")

class WithinPersonCentering(BaseEstimator, TransformerMixin):
    def __init__(self, person_col): 
        self.person_col = person_col

    def fit(self, X, y=None):
        self.numeric_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        if self.person_col in self.numeric_cols_: 
            self.numeric_cols_.remove(self.person_col)
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.numeric_cols_:
            if ITEM_ID not in c:
                X[f"{c}_relative"] = X[c] - X.groupby(self.person_col)[c].transform('mean')
        return X

class CapRT(BaseEstimator, TransformerMixin):
    def __init__(self, quantile=0.99): self.quantile = quantile
    def fit(self, X, y=None):
        rt_cols = [c for c in X.columns if "rt" in c.lower() and "missing" not in c.lower()]
        self.caps_ = X[rt_cols].quantile(self.quantile)
        return self
    def transform(self, X):
        X = X.copy()
        for c, cap in self.caps_.items():
            if c in X.columns: X[c] = X[c].clip(lower=0, upper=cap)
        return X

# ------------------------------------------------------------
# 2) METRICS
# ------------------------------------------------------------
def get_spearman(df, score_col):
    rhos = []
    for _, g in df.groupby(PERSON_ID):
        if len(g) < 3: continue
        yt, yp = g['y_true_c'].values, g[score_col].values
        if np.nanstd(yt) < 1e-12 or np.nanstd(yp) < 1e-12: continue
        rho, _ = spearmanr(yt, yp); rhos.append(rho)
    return np.mean(rhos) if rhos else 0.0

# ------------------------------------------------------------
# 3) MODELING & VISUALIZATION
# ------------------------------------------------------------
df = pd.read_csv(DATA_PATH); y = df[TARGET].astype(float); X = df.drop(columns=[TARGET]).copy()
mlflow.set_experiment("food_liking_no_leakage")

pipeline = Pipeline([
    ("centering", WithinPersonCentering(person_col=PERSON_ID)),
    ("cleaner", BehavioralCleaner(substrings=DROP_SUBSTRINGS, ids_to_kill=ID_COLS)),
    ("capping", CapRT(quantile=RT_QUANTILE)),
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("scaler", StandardScaler()),
])

models = {
    "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=RANDOM_STATE),
    "SVR": SVR(kernel='rbf', C=1.0, epsilon=0.1),
    "Ridge": Ridge(alpha=10.0, random_state=RANDOM_STATE),
    "HistGB": HistGradientBoostingRegressor(random_state=RANDOM_STATE, max_iter=200)
}

cv = GroupKFold(n_splits=N_SPLITS)

for model_name, reg in models.items():
    with mlflow.start_run(run_name=f"{model_name}_Leakage_Free"):
        all_res = []
        for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups=X[ITEM_ID]), 1):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
            
            mu = y_tr.groupby(X_tr[PERSON_ID]).mean()
            y_tr_c, y_va_c = y_tr - X_tr[PERSON_ID].map(mu), y_va - X_va[PERSON_ID].map(mu)

            Xtr_p = pipeline.fit_transform(X_tr)
            Xva_p = pipeline.transform(X_va)
            
            reg.fit(Xtr_p, y_tr_c.fillna(0))
            
            fold_df = pd.DataFrame({
                PERSON_ID: X_va[PERSON_ID], 
                "y_true_c": y_va_c, 
                "y_pred": reg.predict(Xva_p)
            }).dropna()
            all_res.append(fold_df)

        results = pd.concat(all_res)
        results['y_rand'] = results.groupby(PERSON_ID)['y_pred'].transform(np.random.permutation)
        rho, rho_c = get_spearman(results, "y_pred"), get_spearman(results, "y_rand")

        print(f"\n[{model_name}] Spearman: {rho:.4f} vs Chance: {rho_c:.4f}")

        # --- UPDATED PLOTTING ---
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Left Plot: Performance vs Chance
        sns.barplot(x=['Chance', 'Model'], y=[rho_c, rho], ax=axes[0], palette="coolwarm")
        axes[0].set_title(f"{model_name} Performance (Within-Person Spearman ρ)", fontsize=14)
        axes[0].set_ylabel("Correlation Coefficient")

        # Feature Importance logic
        feat_names = pipeline.named_steps['imputer'].get_feature_names_out()
        
        if model_name in ["Ridge", "ElasticNet"]:
            imps = np.abs(reg.coef_)
            imp_title = "Feature Weights (Beta Coefficients)"
        else:
            r = permutation_importance(reg, Xva_p, y_va_c.fillna(0), n_repeats=5, random_state=RANDOM_STATE)
            imps = r.importances_mean
            imp_title = "Permutation Importance"
        
        # Create Importance DF and apply mapping
        fi = pd.DataFrame({'Feature': feat_names, 'Imp': imps}).sort_values('Imp', ascending=False).head(10)
        fi['Feature'] = fi['Feature'].map(lambda x: RENAME_DICT.get(x, x))
        
        # Right Plot: Importance
        sns.barplot(x='Imp', y='Feature', data=fi, ax=axes[1], palette="viridis")
        axes[1].set_title(f"Top 10 Behavioral Predictors of Preference", fontsize=14)
        axes[1].set_xlabel("Impact on Prediction Score")
        axes[1].set_ylabel("Behavioral Metric")

        plt.tight_layout(); plt.show()
