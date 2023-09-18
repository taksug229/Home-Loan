# -------- Basic Settings --------

TARGET_F = "TARGET_BAD_FLAG"
TARGET_A = "TARGET_LOSS_AMT"
targetcols = [TARGET_F, TARGET_A]
objcols = ["REASON", "JOB"]
numcols = [
    TARGET_F,
    TARGET_A,
    "LOAN",
    "MORTDUE",
    "VALUE",
    "YOJ",
    "DEROG",
    "DELINQ",
    "CLAGE",
    "NINQ",
    "CLNO",
    "DEBTINC",
]
figsize = (6, 4)

# -------- Main Settings --------
random_state = 1  # Data split random_state
cv = 5  # Cross Validation K Folds
max_depth = 3  # Depth size for tree
feature_thresh = 0.05  # Feature importance threshold for tree models
max_step_wise_vars = 10  # Max step wise variables

# -------- Grid Search Parameters --------
UseGridSearch = False  # If True, Grid search parameters will be used. If False, Default Parameters will be used.

grid_search_dic = {
    "Decision Tree": {"max_depth": [3, 5], "min_samples_leaf": [1, 2]},
    "Random Forest": {
        "max_depth": [3, 5],
        "n_estimators": [100, 200],
        "bootstrap": [True, False],
    },
    "Gradient Boosting": {
        "max_depth": [3, 5],
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1],
    },
}

if UseGridSearch:
    title_params = "GS-Params"
else:
    title_params = "Default-Params"
