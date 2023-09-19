# -------- Basic Settings --------
import tensorflow as tf

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
random_state = 1  # Data split random_state.
cv = 5  # Cross Validation K Folds.
max_depth = 3  # Depth size for tree models.
feature_thresh = 0.05  # Feature importance threshold for tree models.
max_step_wise_vars = 10  # Max step wise variables.

# -------- Deep Learning Settings --------
feature_thresh_dl = 0.005  # Feature importance threshold for deep learning models.
variable_selection_model = "RF"  # Variable selection for deep learning models. Choose between "DT", "RF", "GB" or None

activations_names = {
    "Relu": tf.keras.activations.relu,
    "Sigmoid": tf.keras.activations.sigmoid,
    "Tanh": tf.keras.activations.tanh,
}
verbose = False
epochs_categorical = 1  # Original 100
epochs_regression = 1  # Original 800
drop_out_ratio = 0.2  # Drop ratio for drop out layer for deep learning models.

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
