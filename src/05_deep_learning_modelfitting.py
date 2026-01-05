"""
Script: 05_deeplearning_modelfitting.py
Project: Motivational Salience Index (MSI)
Author: Marie Pittet
Description: This scripts fits a LSTM network to trial-level behavioral data.
"""
# ------------------------------------------------------------
# 0) Env
# ------------------------------------------------------------
import pandas as pd
import numpy as np
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks # type: ignore
from scipy.stats import spearmanr

# ------------------------------------------------------------
# 1) CONFIGURATION
# ------------------------------------------------------------
TRAIN_PATH = "data/preprocessed/by_trial/training.csv"
TEST_PATH = "data/preprocessed/by_trial/test.csv"

TARGET = "vas_score"
PERSON_ID = "fk_device_id"
ITEM_ID = "item_id"

# Select numeric features only
FEATURE_COLS = ['rt', 'is_correct', 'is_hit', 'is_miss', 'is_cr', 'is_fa', 'is_go']
SEQ_LEN = 30  
MLFLOW_EXPERIMENT = "food_liking_trial_level_lstm"

# ------------------------------------------------------------
# 2) DATA SEQUENCING (With Dtype Fix)
# ------------------------------------------------------------
def create_sequences(path, seq_len):
    df = pd.read_csv(path)
    
    # --- ensuring all features are numeric ---
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    sequences = []
    targets = []
    group_ids = []

    for (pid, item), group in df.groupby([PERSON_ID, ITEM_ID]):
        group = group.sort_values('trial_index')
        
        # Explicitly cast to float32 to avoid 'object' dtype errors
        feats = group[FEATURE_COLS].values.astype('float32')
        
        if len(feats) < seq_len:
            padding = np.zeros((seq_len - len(feats), len(FEATURE_COLS)), dtype='float32')
            feats = np.vstack([padding, feats])
        else:
            feats = feats[-seq_len:]
            
        sequences.append(feats)
        targets.append(group[TARGET].iloc[0])
        group_ids.append(pid)
        
    return np.array(sequences, dtype='float32'), np.array(targets, dtype='float32'), np.array(group_ids)

# ------------------------------------------------------------
# 3) MODEL TRAINING & EVALUATION
# ------------------------------------------------------------
X_train, y_train, pids_train = create_sequences(TRAIN_PATH, SEQ_LEN)
X_test, y_test, pids_test = create_sequences(TEST_PATH, SEQ_LEN)

mlflow.set_experiment(MLFLOW_EXPERIMENT)

with mlflow.start_run():
    model = models.Sequential([
        layers.Input(shape=(SEQ_LEN, len(FEATURE_COLS))),
        # Masking layer tells LSTM to ignore the zero-padding
        layers.Masking(mask_value=0.0),
        layers.LSTM(64, dropout=0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='linear') 
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    mlflow.tensorflow.autolog()

    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    
    model.fit(
        X_train, y_train, 
        epochs=100, 
        batch_size=32, 
        validation_split=0.2, 
        callbacks=[early_stop]
    )

    # Within-person Spearman Evaluation
    preds = model.predict(X_test).flatten()
    eval_df = pd.DataFrame({'pid': pids_test, 'y_true': y_test, 'y_pred': preds})

    rhos = []
    for _, g in eval_df.groupby('pid'):
        if len(g) > 2: # Need at least 3 items to rank
            rho, _ = spearmanr(g['y_true'], g['y_pred'])
            if np.isfinite(rho): rhos.append(rho)
    
    final_rho = np.mean(rhos) if rhos else 0
    mlflow.log_metric("mean_within_person_spearman", final_rho)
    
    print(f"Trial-Level LSTM Mean Spearman: {final_rho:.4f}")
