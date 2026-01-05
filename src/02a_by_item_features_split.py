
"""
Script: 02a_by_item_features_split.py
Project: Motivational Salience Index (MSI)
Author: Marie Pittet
Description: This script further prepares the data for ML analyses by:
    - computing Signal Detection Theory indices
    - pivoting the dataset wide
    - performing the train-test split at that stage (before dealing with missing data and normalizing) to avoid leaking 
"""

# ------------------------------------------------------------
# 0) Env
# ------------------------------------------------------------
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import GroupShuffleSplit

# ------------------------------------------------------------
# 1) Loading the dataset
# ------------------------------------------------------------
df = pd.read_csv("data/extracted/item_df.csv")


# ------------------------------------------------------------
# 2) Computing Signal Detection Theory metrics (maybe more useful that raw hit/misses/etc)
# ------------------------------------------------------------
# counts of hits, miss, false alarms, correct rejections
nh, nm, nfa, ncr = df["n_hit"], df["n_miss"], df["n_fa"], df["n_cr"]

# raw rates (will be NaN if denominator is 0)
df["hit_rate"] = nh / (nh + nm)
df["fa_rate"]  = nfa / (nfa + ncr)

# log-linear correction to prevents norm.ppf(0) or norm.ppf(1) from blowing up to infinity when hit/FA rates are exactly 0 or 1.
Hc = (nh  + 0.5) / (nh  + nm  + 1.0)
Fc = (nfa + 0.5) / (nfa + ncr + 1.0)

# if a row has zero trials (denominator 0), set to NaN
Hc = Hc.where((nh + nm) > 0)
Fc = Fc.where((nfa + ncr) > 0)

# SDT metrics
df["dprime"]    = norm.ppf(Hc) - norm.ppf(Fc)
df["criterion"] = -0.5 * (norm.ppf(Hc) + norm.ppf(Fc))


# ------------------------------------------------------------
# 3) Pivoting wide
# ------------------------------------------------------------
id_cols = ["fk_device_id","item_id"]
y = df.groupby(id_cols)["vas_score"].first()  
feature_cols = [
    "n_trials", "acc_go","acc_nogo", "mean_rt_go", "median_rt_go", 
    "mean_rt_fa", "median_rt_fa",
    "hit_rate","fa_rate","dprime","criterion",
    "n_hit","n_miss","n_cr","n_fa",
]

X_long = df[id_cols + ["task"] + feature_cols].copy()

X_wide = X_long.pivot_table(
    index=id_cols,
    columns="task",
    values=feature_cols,
    aggfunc="mean"
)

# Flatten multiindex columns
X_wide.columns = [f"{feat}_{task}" for feat, task in X_wide.columns]
X_wide = X_wide.reset_index()

# Merge
wide_df = X_wide.merge(y.reset_index(), on=id_cols, how="inner")

# ------------------------------------------------------------
# 4) train-test split
# ------------------------------------------------------------
X = wide_df.drop(columns=["vas_score"])
y = wide_df["vas_score"]
groups = wide_df["fk_device_id"] # Yes I never bothered renaming that as participant_ID

# performing a 70%/30% train-test split
gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=123)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
y_train, y_test = y.iloc[train_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)

# saving that
train_df = X_train.copy()
train_df["vas_score"] = y_train

test_df = X_test.copy()
test_df["vas_score"] = y_test

train_df.to_csv("data/preprocessed/training.csv", index=False)
test_df.to_csv("data/preprocessed/test.csv", index=False)



