"""
Script: 02b_by_trial_split.py
Project: Motivational Salience Index (MSI)
Author: Marie Pittet
Description: This script:
- Does some feature engineering
- Performs participant-level train/test split.
- Saves as long-format CSVs for LSTM/RNN ingestion.
"""
# ------------------------------------------------------------
# 0) Env
# ------------------------------------------------------------wq
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# 1) Loading the raw trial-level dataset
# ------------------------------------------------------------
#this file contains raw trial rows
df = pd.read_csv("../data/extracted/trial_df.csv") 

TARGET = "vas_score"
PERSON_ID = "fk_device_id"
ITEM_ID = "item_id"

# ------------------------------------------------------------
# 2) Participant-Level Train-Test Split (70/30)
# ------------------------------------------------------------
# This ensures a participant's entire history is either in train or test
gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=123)
train_idx, test_idx = next(gss.split(df, groups=df[PERSON_ID]))

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

# ------------------------------------------------------------
# 3) Feature Engineering & Normalization
# ------------------------------------------------------------
# Convert categorical outcomes to numeric flags for the Neural Network
def engineer_trial_features(data):
    # Map trial types and outcomes to numeric
    data['is_go'] = (data['trial_type'] == 'go').astype(int)
    data['is_correct'] = data['is_correct'].astype(int)
    
    # Handle RTs: Fill NaNs (No-Go trials) with 0 or a specific indicator
    data['rt'] = data['rt'].fillna(0)
    
    return data

train_df = engineer_trial_features(train_df)
test_df = engineer_trial_features(test_df)

# ------------------------------------------------------------
# 4) Saving the Datasets
# ------------------------------------------------------------
train_df.to_csv("../data/preprocessed/by_trial/training.csv", index=False)
test_df.to_csv("../data/preprocessed/by_trial/test.csv", index=False)
