# -*- coding: utf-8 -*-
"""
Script: 03_ML_modelfitting_testing.py
Description: Benchmarks ElasticNet, Ridge, and HistGB, and evaluates on held-out test data.
- We report both model performance and a chance baseline.
- Chance is computed by permuting predictions within each person (destroys item-level mapping).
"""

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
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

# ------------------------------------------------------------
# 1) Configs, paths, dicts
# ------------------------------------------------------------
TRAIN_PATH = "../data/preprocessed/by_item/training.csv"
TEST_PATH  = "../data/preprocessed/by_item/test.csv"

TARGET, PERSON_ID, ITEM_ID = "vas_score", "fk_device_id", "item_id"
ID_COLS = [PERSON_ID, ITEM_ID]

DROP_SUBSTRINGS = ["SST", "n_trials"]
N_SPLITS, RT_QUANTILE, RANDOM_STATE = 5, 0.99, 42

RENAME_DICT = {
    'n_fa_GNG_relative': 'Inhibitory Failure (GNG FA)',
    'mean_rt_go_GNG_relative': 'Approach Speed (GNG Go RT)',
    'acc_nogo_CAT_relative': 'Self-Control Accuracy (CAT)',
}

# ------------------------------------------------------------
# 2) transformers
# ------------------------------------------------------------
class BehavioralCleaner(BaseEstimator, TransformerMixin):
    """Removes all non-behavioral IDs and substrings to prevent leakage."""
    def __init__(self, substrings, ids_to_kill):
        self.substrings = substrings
        self.ids_to_kill = ids_to_kill

    def fit(self, X, y=None):
        cols = list(X.columns)
        self.cols_to_drop_ = [
            c for c in cols
            if any(s in c for s in self.substrings)
            or any(i in c for i in self.ids_to_kill)
        ]
        return self

    def transform(self, X):
        return X.drop(columns=self.cols_to_drop_, errors="ignore")


class WithinPersonCentering(BaseEstimator, TransformerMixin):
    """Creates *_relative features: each behavioral metric minus that person's mean."""
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
                X[f"{c}_relative"] = X[c] - X.groupby(self.person_col)[c].transform("mean")
        return X


class CapRT(BaseEstimator, TransformerMixin):
    """Caps RT variables at a robust quantile (fit on training fold only)."""
    def __init__(self, quantile=0.99):
        self.quantile = quantile

    def fit(self, X, y=None):
        rt_cols = [c for c in X.columns if "rt" in c.lower() and "missing" not in c.lower()]
        self.caps_ = X[rt_cols].quantile(self.quantile) if rt_cols else pd.Series(dtype=float)
        return self

    def transform(self, X):
        X = X.copy()
        for c, cap in self.caps_.items():
            if c in X.columns:
                X[c] = X[c].clip(lower=0, upper=cap)
        return X

# ------------------------------------------------------------
# 3) Metrics, utils
# ------------------------------------------------------------
def get_spearman(df, target_col, pred_col):
    """Within-person Spearman averaged across persons."""
    rhos = []
    for _, g in df.groupby(PERSON_ID):
        if len(g) < 3:
            continue
        yt, yp = g[target_col].values, g[pred_col].values
        if np.nanstd(yt) < 1e-12 or np.nanstd(yp) < 1e-12:
            continue
        rho, _ = spearmanr(yt, yp)
        rhos.append(rho)
    return np.mean(rhos) if rhos else 0.0


def add_within_person_chance(df, pred_col="y_pred", out_col="y_rand", seed=RANDOM_STATE):
    """
    Chance baseline: permute predictions within each person.
    Preserves each person's prediction distribution but breaks item mapping.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    def _permute(arr):
        arr = np.asarray(arr)
        return rng.permutation(arr)

    df[out_col] = df.groupby(PERSON_ID)[pred_col].transform(_permute)
    return df

# ------------------------------------------------------------
# 4) Execution loop
# ------------------------------------------------------------
df_train = pd.read_csv(TRAIN_PATH)
df_test  = pd.read_csv(TEST_PATH)

mlflow.set_experiment("food_liking_final_test")

pipeline = Pipeline([
    ("centering", WithinPersonCentering(person_col=PERSON_ID)),
    ("cleaner", BehavioralCleaner(substrings=DROP_SUBSTRINGS, ids_to_kill=ID_COLS)),
    ("capping", CapRT(quantile=RT_QUANTILE)),
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("scaler", StandardScaler()),
])

models = {
    "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=RANDOM_STATE),
    "Ridge": Ridge(alpha=10.0, random_state=RANDOM_STATE),
    "HistGB": HistGradientBoostingRegressor(random_state=RANDOM_STATE, max_iter=200),
}

cv = GroupKFold(n_splits=N_SPLITS)

for model_name, reg in models.items():
    with mlflow.start_run(run_name=f"{model_name}_Final_Eval"):

        # ------------------------------------------------------------
        # A) Cross-validation (grouped by item)
        # ------------------------------------------------------------
        X = df_train.drop(columns=[TARGET])
        y = df_train[TARGET].astype(float)

        all_cv_res = []

        for tr_idx, va_idx in cv.split(X, y, groups=X[ITEM_ID]):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            # Within-person centering of TARGET to remove leniency / response bias
            mu = y_tr.groupby(X_tr[PERSON_ID]).mean()
            y_tr_c = (y_tr - X_tr[PERSON_ID].map(mu)).fillna(0)
            y_va_c = (y_va - X_va[PERSON_ID].map(mu)).fillna(0)

            # Fold-wise preprocessing
            Xtr_p = pipeline.fit_transform(X_tr)
            Xva_p = pipeline.transform(X_va)

            reg.fit(Xtr_p, y_tr_c)

            fold_df = pd.DataFrame({
                PERSON_ID: X_va[PERSON_ID],
                "y_true": y_va_c,
                "y_pred": reg.predict(Xva_p),
            })
            all_cv_res.append(fold_df)

        cv_results = pd.concat(all_cv_res, ignore_index=True)

        # Model performance (CV)
        rho_cv = get_spearman(cv_results, "y_true", "y_pred")

        # Chance baseline (CV)
        cv_results = add_within_person_chance(cv_results, pred_col="y_pred", out_col="y_rand", seed=RANDOM_STATE)
        rho_cv_chance = get_spearman(cv_results, "y_true", "y_rand")

        # ------------------------------------------------------------
        # B) Generalization performance on held-out test set
        # ------------------------------------------------------------
        # Train on all training data
        mu_full = y.groupby(X[PERSON_ID]).mean()
        y_train_c = (y - X[PERSON_ID].map(mu_full)).fillna(0)

        X_train_full_p = pipeline.fit_transform(X)
        reg.fit(X_train_full_p, y_train_c)

        # Predict test
        X_test = df_test.drop(columns=[TARGET])
        y_test = df_test[TARGET].astype(float)

        # Center test target
        mu_test = y_test.groupby(X_test[PERSON_ID]).mean()
        y_test_c = (y_test - X_test[PERSON_ID].map(mu_test)).fillna(0)

        X_test_p = pipeline.transform(X_test)
        y_test_pred = reg.predict(X_test_p)

        test_df = pd.DataFrame({
            PERSON_ID: X_test[PERSON_ID],
            "y_true": y_test_c,
            "y_pred": y_test_pred,
        })

        rho_test = get_spearman(test_df, "y_true", "y_pred")

        # Chance baseline
        test_df = add_within_person_chance(test_df, pred_col="y_pred", out_col="y_rand", seed=RANDOM_STATE)
        rho_test_chance = get_spearman(test_df, "y_true", "y_rand")

        print(
            f"\n[{model_name}] "
            f"CV Spearman: {rho_cv:.4f} (Chance: {rho_cv_chance:.4f}) | "
            f"TEST Spearman: {rho_test:.4f} (Chance: {rho_test_chance:.4f})"
        )

        # ------------------------------------------------------------
        # C) Visualization
        # ------------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Performance bars: model vs chance, CV vs TEST
        sns.barplot(
            x=["CV Chance", "CV Model", "TEST Chance", "TEST Model"],
            y=[rho_cv_chance, rho_cv, rho_test_chance, rho_test],
            ax=axes[0],
            palette="magma",
        )
        axes[0].set_title(f"{model_name}: Generalization + Chance Benchmark")
        axes[0].set_ylabel("Within-Person Spearman ρ")
        axes[0].set_xlabel("")

        # Feature importance (final model, evaluated on TEST inputs)
        feat_names = pipeline.named_steps["imputer"].get_feature_names_out()

        if hasattr(reg, "coef_"):
            imps = np.abs(reg.coef_)
        else:
            perm = permutation_importance(
                reg,
                X_test_p,
                y_test_c,
                n_repeats=5,
                random_state=RANDOM_STATE,
            )
            imps = perm.importances_mean

        fi = (
            pd.DataFrame({"Feature": feat_names, "Imp": imps})
            .sort_values("Imp", ascending=False)
            .head(10)
        )
        fi["Feature"] = fi["Feature"].map(lambda x: RENAME_DICT.get(x, x))

        sns.barplot(x="Imp", y="Feature", data=fi, ax=axes[1], palette="viridis")
        axes[1].set_title("Top Predictors (Final Model)")
        axes[1].set_xlabel("Importance")
        axes[1].set_ylabel("")

        plt.tight_layout()
        plt.show()

        # ------------------------------------------------------------
        # D) MLflow logging
        # ------------------------------------------------------------
        mlflow.log_metric("rho_cv", rho_cv)
        mlflow.log_metric("rho_cv_chance", rho_cv_chance)
        mlflow.log_metric("rho_test", rho_test)
        mlflow.log_metric("rho_test_chance", rho_test_chance)
        mlflow.sklearn.log_model(reg, f"model_{model_name}")