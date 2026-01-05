"""
Script: 03_by_item_imputation_normalization.py
Project: Motivational Salience Index (MSI)
Author: Marie Pittet
Description: This script:
- Performs Within subject preference centering
- Performs Capping, scaling, and imputation
"""

# ------------------------------------------------------------
# 0) Env
# ------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

# ------------------------------------------------------------
# 1) Loading data (Keep IDs for grouping)
# ------------------------------------------------------------
train = pd.read_csv("data/preprocessed/by_item/training.csv")
test  = pd.read_csv("data/preprocessed/by_item/test.csv")

TARGET = "vas_score"
ID_COLS = ["fk_device_id", "item_id"]

# ------------------------------------------------------------
# 2) Drop SST + n_trials
# ------------------------------------------------------------
drop_patterns = ["SST", "n_trials"]
cols_to_drop = [c for c in train.columns if any(p in c for p in drop_patterns)]
train.drop(columns=cols_to_drop, inplace=True)
test.drop(columns=cols_to_drop, inplace=True, errors="ignore")

# ------------------------------------------------------------
# 3) Within-Subject Centering (The "Preference" Signal)
# ------------------------------------------------------------
# This captures if a person was faster/slower than THEIR OWN average for an item.
rt_cols = [c for c in train.columns if "rt" in c.lower() and c not in ID_COLS]

for c in rt_cols:
    # Calculate per-person mean on train
    train_means = train.groupby("fk_device_id")[c].transform("mean")
    # Apply to train
    train[f"{c}_relative"] = train[c] - train_means
    
    # For test, we ideally use the train_means per person if available, 
    # but for a general pipeline, we use the test person's own mean.
    test_means = test.groupby("fk_device_id")[c].transform("mean")
    test[f"{c}_relative"] = test[c] - test_means

# ------------------------------------------------------------
# 4) RT Capping & Transformation
# ------------------------------------------------------------
rt_relative_cols = [c for c in train.columns if "_relative" in c]
all_rt_features = rt_cols + rt_relative_cols

# Learn caps from TRAIN only (original RTs)
rt_caps = train[rt_cols].quantile(0.99)

for c in rt_cols:
    cap = rt_caps[c]
    train[c] = train[c].clip(lower=0, upper=cap)
    test[c]  = test[c].clip(lower=0, upper=cap)
    
    # We keep raw RTs and relative RTs. 
    # Log-transform is optional; if using XGBoost, raw is often better.
    # We'll stick to raw for now as relative RT handles the variance.

# ------------------------------------------------------------
# 5) Scaling & Imputation Pipeline
# ------------------------------------------------------------
# Separate features from target/IDs
X_train = train.drop(columns=[TARGET] + ID_COLS)
X_test  = test.drop(columns=[TARGET] + ID_COLS)
y_train = train[TARGET]
y_test  = test[TARGET]

# Impute with median + add indicators
imputer = SimpleImputer(strategy="median", add_indicator=True)
X_train_imp = imputer.fit_transform(X_train)
X_test_imp  = imputer.transform(X_test)

imp_cols = imputer.get_feature_names_out()

# Identify which columns to scale (don't scale the binary 0/1 indicators)
indicator_mask = ["missingindicator" in c for c in imp_cols]
orig_mask = [not m for m in indicator_mask]

scaler = StandardScaler()
X_train_scaled_part = scaler.fit_transform(X_train_imp[:, orig_mask])
X_test_scaled_part  = scaler.transform(X_test_imp[:, orig_mask])

# Recombine
X_train_final = pd.DataFrame(
    np.hstack([X_train_scaled_part, X_train_imp[:, indicator_mask]]),
    columns=list(np.array(imp_cols)[orig_mask]) + list(np.array(imp_cols)[indicator_mask])
)

X_test_final = pd.DataFrame(
    np.hstack([X_test_scaled_part, X_test_imp[:, indicator_mask]]),
    columns=list(np.array(imp_cols)[orig_mask]) + list(np.array(imp_cols)[indicator_mask])
)

# ------------------------------------------------------------
# 6) Save & Export
# ------------------------------------------------------------
joblib.dump({"imputer": imputer, "scaler": scaler}, "preprocess_v2.joblib")

# Include IDs back in for Group-KFold cross-validation later
X_train_final["fk_device_id"] = train["fk_device_id"].values
X_train_final[TARGET] = y_train.values

X_test_final["fk_device_id"] = test["fk_device_id"].values
X_test_final[TARGET] = y_test.values

X_train_final.to_csv("training_processed.csv", index=False)
X_test_final.to_csv("test_processed.csv", index=False)