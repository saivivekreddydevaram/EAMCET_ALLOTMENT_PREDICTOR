# EAMCET Allotment Predictor

A machine learning project that predicts EAMCET (Engineering, Agriculture and Medicine Common Entrance Test) college cutoff ranks based on historical allotment data, to help students estimate which colleges and branches they may be eligible for based on their rank, gender, and caste category.

## Overview

The project uses historical EAMCET allotment data (college-wise, branch-wise cutoff ranks across caste and gender categories) to train a Random Forest Regressor that predicts cutoff ranks for a given college, branch, category, and year combination.

## Data Source

Raw data: [EMCET.csv](https://raw.githubusercontent.com/saivivekreddydevaram/EAMCET_ALLOTMENT_PREDICTOR/refs/heads/main/EMCET.csv)

Each row in the raw data represents one college + branch combination, with separate columns for cutoff ranks across 18 category/gender combinations (e.g. `OCB`, `OCG`, `BCAB`, `SCB`, `EWSG`, etc.), plus metadata like institute name, place, district, tuition fee, and affiliating university.

## Project Structure

### 1. Preprocessing (`EamcetPreprocessor` class)

Converts the wide-format raw CSV into a clean, ML-ready dataset:

- **Reshaping**: Uses `pd.melt` to unpivot the 18 category columns into two columns — `CATEGORY_CODE` and `CUTOFF_RANK` — so each row represents a single college/branch/category/gender cutoff.
- **Feature extraction**: Splits `CATEGORY_CODE` (e.g. `'OCB'`) into `GENDER` (`Boys`/`Girls`) and `CASTE_CATEGORY` (`OC`, `BCA`, `SC`, etc.).
- **Cleaning**: Converts `CUTOFF_RANK` and `TUITION_FEE` to numeric, coercing invalid values to `NaN`.
- **Missing tuition fees**: Imputed using the median fee within the same institute (`INST_CODE`), with a global median fallback and a `TUITION_FEE_MISSING` flag column so the model can account for imputed values.
- **Encoding**: All categorical columns (`INST_CODE`, `PLACE`, `DIST`, `COED`, `TYPE`, `BRANCH`, `AFFILIATED`, `GENDER`, `CASTE_CATEGORY`) are converted to numeric IDs using `LabelEncoder`, with each fitted encoder saved in `self.encoders` for reuse on new prediction inputs.
- **Readable copy**: An unencoded copy of the processed data is kept in `self.readable_df` for filtering and display purposes.

Key methods:
- `transform()` — runs the full preprocessing pipeline, returns the ML-ready dataframe.
- `encode_input(new_input)` — encodes a single raw input dict using the same fitted encoders, for making new predictions.

### 2. Model Training

- **Algorithm**: `RandomForestRegressor` (scikit-learn)
- **Target**: `CUTOFF_RANK`
- **Features**: `INST_CODE`, `PLACE`, `DIST`, `COED`, `TYPE`, `BRANCH`, `AFFILIATED`, `GENDER`, `CASTE_CATEGORY`, `TUITION_FEE`, `YEAR`
- **Split**: 80/20 train/test split via `train_test_split`
- **Target scaling**: Given the wide range of cutoff ranks (double digits up to ~170,000), a log-transform (`np.log1p`) of the target is used during training to weight relative error over absolute error, then reversed with `np.expm1` for predictions.

### 3. Prediction

`predict_cutoff(new_input, preprocessor, rf, feature_cols)` takes a raw dictionary describing a college/branch/category/gender/year combination, encodes it via the preprocessor, and returns the model's predicted cutoff rank.

### 4. Evaluation

Model performance is assessed using:
- **MAE** (Mean Absolute Error) — average absolute rank error
- **RMSE** (Root Mean Squared Error) — penalizes large individual misses more heavily

## Usage

```python
preprocessor = EamcetPreprocessor()
ml_ready_df = preprocessor.transform()

# train/test split + model training
# ...

new_input = {
    'INST_CODE': 'ACEG',
    'PLACE': 'GHATKESAR',
    'DIST': 'MDL',
    'COED': 'COED',
    'TYPE': 'PVT',
    'BRANCH': 'CSE',
    'AFFILIATED': 'JNTUH',
    'GENDER': 'Boys',
    'CASTE_CATEGORY': 'OC',
    'TUITION_FEE': 82000,
    'YEAR': 2019
}

predicted_rank = predict_cutoff(new_input, preprocessor, rf, feature_cols)
print(f"Predicted cutoff rank: {predicted_rank:.0f}")
```

## Planned / In Progress

- **Gender and caste filter**: A lookup function to filter `preprocessor.readable_df` for eligible colleges given a student's rank, gender, and caste category, ranked by closeness of cutoff.
- **Further tuning**: Hyperparameter search (`GridSearchCV`) and feature importance analysis to reduce MAE further, particularly for lower (more competitive) rank ranges where prediction accuracy matters most.

## Requirements

- `pandas`
- `numpy`
- `scikit-learn`

## Notes

- Some institutes report an identical cutoff rank across all categories for a branch (e.g. capped at intake capacity) — this reflects low demand for that seat, not a data error.
- `EWS` category cutoffs are only available from the year EWS reservation was introduced, so historical rows before that year will not have this category and are dropped during preprocessing (rows with `NaN` cutoffs are removed).
