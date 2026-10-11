import joblib
import numpy as np
import pandas as pd

_art = joblib.load("recommender_data.pkl")
latest = _art["latest"]
SAFE_M, TARGET_M, REACH_M = _art["safe_m"], _art["target_m"], _art["reach_m"]
year_max_latest = _art["year_max_latest"]
latest_year = _art["latest_year"]


def recommend(rank, caste, gender, branches=None, districts=None, shift=0.0, show_unlikely=False):
    seats = latest[(latest["CASTE"] == caste) & (latest["GENDER"] == gender)].copy()
    if branches:
        seats = seats[seats["BRANCH"].isin(branches)]
    if districts:
        seats = seats[seats["DIST"].isin(districts)]
    seats["EXPECTED_CUTOFF"] = (seats["CUTOFF_RANK"] * np.exp(shift)).round()
    seats["M"] = rank / seats["EXPECTED_CUTOFF"]

    def label(r):
        if r["CAPPED"]:
            return "Open (likely available)" if rank <= year_max_latest else "Unlikely"
        if r["M"] <= SAFE_M:
            return "Safe"
        if r["M"] <= TARGET_M:
            return "Target"
        if r["M"] <= REACH_M:
            return "Reach"
        return "Unlikely"

    seats["BUCKET"] = seats.apply(label, axis=1)
    if not show_unlikely:
        seats = seats[seats["BUCKET"] != "Unlikely"]
    order = {"Safe": 0, "Target": 1, "Reach": 2, "Open (likely available)": 3}
    seats["_o"] = seats["BUCKET"].map(order)
    seats = seats.sort_values(["_o", "CUTOFF_RANK"])
    return seats[["INST_NAME", "PLACE", "DIST", "BRANCH", "CUTOFF_RANK", "EXPECTED_CUTOFF", "BUCKET"]].reset_index(drop=True)
