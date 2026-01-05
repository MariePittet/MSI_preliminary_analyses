"""
Script: 01_data_extraction.py
Project: Motivational Salience Index (MSI)
Author: Marie Pittet
Description: Turns event-level task logs into:
1) trial_df: one row per trial
2) item_df: one row per food item per task 
- Merges VAS liking score  for each food item 
"""
# ------------------------------------------------------------
# 0) Env
# ------------------------------------------------------------

import pyreadr
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# 1) Load data
# ------------------------------------------------------------
rdata = pyreadr.read_r("data/raw/diner_food_flat.RData")
df_raw = rdata["data"].copy()
df_vas = rdata["calibration"].copy()


# ------------------------------------------------------------
# 2) Event code lookup + task rules
# ------------------------------------------------------------
event_lookup = {
    # GNG
    101: ("GNG_HIT_ON_GO", "Hit"),
    102: ("GNG_NO_GO_NOT_DRAGGED", "Correct Rejection"),
    103: ("GNG_TOO_SLOW_ON_GO", "Hit slow"),
    104: ("GNG_GO_NOT_DRAGGED", "Miss"),
    105: ("GNG_ON_TARGET_ON_NO_GO", "False Alarm"),
    106: ("GNG_STOP_DRAG_BEFORE_TARGET_ON_GO", "Hit"),
    107: ("GNG_STOP_DRAG_BEFORE_TARGET_ON_NO_GO", "False Alarm"),
    108: ("GNG_ELEMENT_READY_FOR_INTERACTION", "Item onset"),
    110: ("GNG_START_DRAG_OBJECT", None),
    111: ("GNG_DRAG_OBJECT", None),
    112: ("GNG_STOP_DRAG_OBJECT", None),
    113: ("GNG_CUE_TYPE", None),

    # SST
    201: ("SST_CLICK_OBJECT_BEFORE_RTT_WRONG_SIDE", "Hit"),
    202: ("SST_CLICK_OBJECT_AFTER_RTT", "Hit slow"),
    203: ("SST_CLICK_OBJECT_BEFORE_STOP_SIGNAL", "Error"),
    204: ("SST_CLICK_OBJECT_AFTER_STOP_SIGNAL", "False Alarm"),
    205: ("SST_DONT_CLICK_OBJECT", "Miss"),
    206: ("SST_DONT_CLICK_OBJECT_ON_STOP_SIGNAL", "Correct Rejection"),
    207: ("SST_CLICK_OBJECT_BEFORE_RTT_GOOD_SIDE", "Hit"),
    208: ("SST_ELEMENT_READY_FOR_INTERACTION", "Item onset"),
    209: ("SST_SHOW_STOP", "Stop signal"),

    # CAT
    301: ("CAT_CLICK_OBJECT_AFTER_CUE_BEFORE_RTT", "Hit"),
    302: ("CAT_CLICK_OBJECT_AFTER_CUE_AFTER_RTT", "Hit slow"),
    303: ("CAT_DONT_CLICK_OBJECT_ON_CUE", "Miss"),
    304: ("CAT_CLICK_OBJECT_BEFORE_CUE", "False Alarm"),
    305: ("CAT_CLICK_OBJECT_ON_NO_CUE", "False Alarm"),
    306: ("CAT_DONT_CLICK_OBJECT_ON_NO_CUE", "Correct Rejection"),
    307: ("CAT_SHOW_CUE", "Stop signal"),
    308: ("CAT_ELEMENT_READY_FOR_INTERACTION", "Item onset"),
    309: ("CAT_LIFE_TIME", "Item offset"),
}

ONSET_CODE = {"GNG": 108, "SST": 208, "CAT": 308}

RESPONSE_CODES = {
    "GNG": [101, 102, 103, 104, 105, 106, 107],
    "SST": [201, 202, 203, 204, 205, 206, 207],
    "CAT": [301, 302, 303, 304, 305, 306],
}
ALL_RESPONSE_CODES = sum(RESPONSE_CODES.values(), [])

KEEP_OUTCOMES = {"Hit", "Hit slow", "Miss", "Correct Rejection", "False Alarm"}  # what you want in the end


# ------------------------------------------------------------
# 3) Clean + label the raw log
# ------------------------------------------------------------
df_raw["event_name"] = df_raw["type"].map(lambda x: event_lookup.get(x, (None, None))[0])
df_raw["outcome"]    = df_raw["type"].map(lambda x: event_lookup.get(x, (None, None))[1])

# task from numeric range
df_raw["task"] = np.select(
    [df_raw["type"].between(100, 199), df_raw["type"].between(200, 299), df_raw["type"].between(300, 399)],
    ["GNG", "SST", "CAT"],
    default="OTHER"
)

# useful numeric parse
df_raw["info_num"] = pd.to_numeric(df_raw["info"], errors="coerce")
df_raw["time"]     = pd.to_numeric(df_raw["time"], errors="coerce")

# always sort before building trials
df_raw = df_raw.sort_values(["fk_device_id", "task", "time"]).reset_index(drop=True)


# ------------------------------------------------------------
# 4) Build trials: trial_index increments at each onset
# ------------------------------------------------------------
df_raw["is_onset"] = df_raw["type"].eq(df_raw["task"].map(ONSET_CODE))

df_raw["trial_index"] = (
    df_raw.groupby(["fk_device_id", "task"])["is_onset"]
          .cumsum()
          .astype(int)
)

# drop rows before the first onset for each device/task
df_raw = df_raw[df_raw["trial_index"] > 0].copy()

# item_id is stored in info on onset rows; propagate within trial
df_raw["item_id"] = np.where(df_raw["is_onset"], df_raw["info_num"], np.nan)
df_raw["item_id"] = df_raw.groupby(["fk_device_id", "task", "trial_index"])["item_id"].transform("max")


# ------------------------------------------------------------
# 5) Trial-level table: onset time + first response + RT
# ------------------------------------------------------------
# onset time per trial
onset_time = (
    df_raw[df_raw["is_onset"]]
    .groupby(["fk_device_id", "task", "trial_index"])["time"]
    .min()
    .rename("t_onset")
)

# response rows: only response codes, only outcomes you care about
resp = df_raw[df_raw["type"].isin(ALL_RESPONSE_CODES)].copy()

# If you want to treat SST "Error" as False Alarm, uncomment:
# resp.loc[resp["outcome"] == "Error", "outcome"] = "False Alarm"

resp = resp[resp["outcome"].isin(KEEP_OUTCOMES)]

# first response event per trial
first_resp = (
    resp.sort_values("time")
        .groupby(["fk_device_id", "task", "trial_index"])
        .first()[["time", "outcome", "type", "event_name"]]
        .rename(columns={"time": "t_response", "type": "response_code", "event_name": "response_event"})
)

# collapse "Hit slow" into "Hit"
first_resp["outcome"] = first_resp["outcome"].replace({"Hit slow": "Hit"})

# assemble trial_df
trial_df = (
    pd.concat([onset_time, first_resp], axis=1)
      .reset_index()
)

# attach item_id
trial_items = (
    df_raw.groupby(["fk_device_id", "task", "trial_index"])["item_id"]
          .max()
          .reset_index()
)

trial_df = trial_df.merge(trial_items, on=["fk_device_id", "task", "trial_index"], how="left")

# RT = response - onset
trial_df["rt"] = trial_df["t_response"] - trial_df["t_onset"]


# ------------------------------------------------------------
# 6) VAS liking: clean + merge + drop missing
# ------------------------------------------------------------
df_vas = df_vas.rename(columns={"imageUID": "item_id", "score": "vas_score"})

df_vas["fk_device_id"] = pd.to_numeric(df_vas["fk_device_id"], errors="coerce").astype("Int64")
df_vas["item_id"]      = pd.to_numeric(df_vas["item_id"], errors="coerce").astype("Int64")
df_vas["vas_score"]    = pd.to_numeric(df_vas["vas_score"], errors="coerce")

# if the same item appears more than once, average it
df_vas = (
    df_vas.groupby(["fk_device_id", "item_id"], as_index=False)["vas_score"]
          .mean()
)

# merge onto trial_df and drop trials without VAS
trial_df = trial_df.merge(df_vas, on=["fk_device_id", "item_id"], how="left")
trial_df = trial_df.dropna(subset=["vas_score"]).copy()
trial_df = trial_df.dropna(subset=["outcome"]).copy()

# ------------------------------------------------------------
# 7) Item-level summary: separate GO vs NOGO + RT only on responses
# ------------------------------------------------------------

# Basic outcome flags
trial_df["is_hit"]  = (trial_df["outcome"] == "Hit")
trial_df["is_miss"] = (trial_df["outcome"] == "Miss")
trial_df["is_cr"]   = (trial_df["outcome"] == "Correct Rejection")
trial_df["is_fa"]   = (trial_df["outcome"] == "False Alarm")

# GO vs NOGO "trial type" from SDT outcomes (optional, but useful)
trial_df["trial_type"] = pd.NA
trial_df.loc[trial_df["is_hit"] | trial_df["is_miss"], "trial_type"] = "go"
trial_df.loc[trial_df["is_cr"]  | trial_df["is_fa"],   "trial_type"] = "nogo"

# Accuracy per trial (still fine)
trial_df["is_correct"] = trial_df["is_hit"] | trial_df["is_cr"]

# RTs ONLY where a response was made
# - GO RT: use hits (response on go)
# - NOGO RT: use false alarms (response on nogo) [optional]
trial_df["rt_go"] = trial_df["rt"].where(trial_df["is_hit"], np.nan)
trial_df["rt_fa"] = trial_df["rt"].where(trial_df["is_fa"],  np.nan)

# Group key
g = ["fk_device_id", "task", "item_id"]

# Aggregate counts + RT summaries
item_df = (
    trial_df.dropna(subset=["item_id"])
    .groupby(g, as_index=False)
    .agg(
        vas_score=("vas_score", "first"),
        n_trials=("trial_index", "count"),

        n_hit=("is_hit", "sum"),
        n_miss=("is_miss", "sum"),
        n_cr=("is_cr", "sum"),
        n_fa=("is_fa", "sum"),

        # old overall acc if you still want it:
        acc=("is_correct", "mean"),

        # RT summaries (response-only)
        mean_rt_go=("rt_go", "mean"),
        median_rt_go=("rt_go", "median"),

        # optional: RT on false alarms
        mean_rt_fa=("rt_fa", "mean"),
        median_rt_fa=("rt_fa", "median"),
    )
)

# Compute go/nogo accuracies from counts (vectorized, no lambdas)
go_den = item_df["n_hit"] + item_df["n_miss"]
nogo_den = item_df["n_cr"] + item_df["n_fa"]

item_df["acc_go"] = item_df["n_hit"] / go_den
item_df.loc[go_den == 0, "acc_go"] = np.nan

item_df["acc_nogo"] = item_df["n_cr"] / nogo_den
item_df.loc[nogo_den == 0, "acc_nogo"] = np.nan

# ------------------------------------------------------------
# 8) Extracting the dataframes for later use
# ------------------------------------------------------------
trial_df.to_csv("data/extracted/trial_df.csv", index=False)
item_df.to_csv("data/extracted/item_df.csv", index=False)
