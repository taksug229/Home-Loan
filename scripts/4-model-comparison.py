import os
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Union
from sklearn.base import BaseEstimator
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, mean_squared_error
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


# -------- Settings --------
random_state = 1  # Data split random_state
figsize = (6, 4)  # Figure size for plots

# -------- Model Settings --------
model_dic = {
    "LR": {
        "Categorical": LogisticRegression(),
        "Regression": LinearRegression(),
    },
    "DT": {
        "Categorical": tree.DecisionTreeClassifier(),
        "Regression": tree.DecisionTreeRegressor(),
    },
    "RF": {
        "Categorical": RandomForestClassifier(),
        "Regression": RandomForestRegressor(),
    },
    "GB": {
        "Categorical": GradientBoostingClassifier(),
        "Regression": GradientBoostingRegressor(),
    },
}
max_depth = 3  # Max depth size for tree
feature_thresh = 0.005  # Feature importance threshold for tree models
variable_selection_model = None  # Choose between "DT", "RF", "GB" or None

# -------- TensorFlow Model Settings --------
activations_names = {
    "Relu": tf.keras.activations.relu,
    "Sigmoid": tf.keras.activations.sigmoid,
    "Tanh": tf.keras.activations.tanh,
}
verbose = False
epochs_categorical = 100
epochs_regression = 800


def get_units(size: int) -> int:
    """Creates unit size for neural network.

    Args:
        size (int): Variable size.

    Returns:
        int: Unit size
    """
    return int(2 * size)


drop_out_ratio = 0.2

# -------- Basic Setups and Functions --------
path = os.path.dirname(os.path.realpath(__file__))
if not variable_selection_model:
    variable_selection_model = "All"
img_path = path + f"/img/{variable_selection_model}_vars/"
log_path = path + f"/logs/{variable_selection_model}_vars/"
os.makedirs(img_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)
current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
log_file_path = (
    log_path
    + f"analysis-log-{formatted_datetime}--var-{variable_selection_model}-rs{random_state}-md{max_depth}-ft{feature_thresh}.log"
)
logging.basicConfig(
    level=logging.INFO,
    filename=log_file_path,
    filemode="w",
    format="%(asctime)s - %(levelname)s: %(message)s",
)
logger = logging.getLogger()


def get_important_features(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    Y_train: pd.Series,
) -> list:
    """
    Train and evaluate a machine learning model.

    Args:
        model (BaseEstimator): The machine learning model from sklearn to be trained and evaluated.
        X_train (pd.DataFrame): Features of the training data.
        Y_train (pd.Series): Target variables of the training data.

    Returns:
        list: Column list containing important.
    """
    global random_state, feature_thresh
    model.set_params(**{"max_depth": max_depth, "random_state": random_state})
    model_ = model.fit(X_train, Y_train)
    importance_feat = X_train.columns[model_.feature_importances_ > feature_thresh]
    return list(importance_feat)


@ignore_warnings(category=ConvergenceWarning)
def process_model(
    model: BaseEstimator,
    model_name: str,
    X_train: pd.DataFrame,
    Y_train: pd.Series,
    X_test: pd.DataFrame,
    Y_test: pd.Series,
    isCat: bool,
    theEpochs: Union[None, int] = None,
) -> Tuple[object, pd.DataFrame]:
    """
    Train and evaluate a machine learning model with optional Grid Search.

    Args:
        model (BaseEstimator): The machine learning model to be trained and evaluated.
        model_name (str): Name of the model.
        X_train (pd.DataFrame): Features of the training data.
        Y_train (pd.Series): Target variables of the training data.
        X_test (pd.DataFrame): Features of the test data.
        Y_test (pd.Series): Target variables of the test data.
        isCat (bool): True if the problem is a classification task, False if regression.
        theEpochs (Union[None, int], optional): The epochs for TensorFlow models.

    Returns:
        Tuple[object, pd.DataFrame]: A tuple containing the trained model, result DataFrame with evaluation metrics,
            cross-validation scores, and DataFrame containing feature importances.
    """

    global verbose
    if theEpochs:
        model.fit(X_train, Y_train, epochs=theEpochs, verbose=verbose)
        if isCat:
            Y_Pred_train = np.argmax(model.predict(X_train), axis=1)
            Y_Pred_test = np.argmax(model.predict(X_test), axis=1)
        else:
            Y_Pred_train = model.predict(X_train)
            Y_Pred_test = model.predict(X_test)
    else:
        model.fit(X_train, Y_train)
        Y_Pred_train = model.predict(X_train)
        Y_Pred_test = model.predict(X_test)
    if isCat:
        train_accuracy = accuracy_score(Y_train, Y_Pred_train)
        test_accuracy = accuracy_score(Y_test, Y_Pred_test)
        logger.info(f"{model_name} Accuracy Train: {train_accuracy:.4f}")
        logger.info(f"{model_name} Accuracy Test: {test_accuracy:.4f}")

        def create_results(X, Y, data_type):
            if theEpochs:
                probs = model.predict(X)
            else:
                probs = model.predict_proba(X)
            p = probs[:, 1]
            fpr, tpr, _ = roc_curve(Y, p)
            roc_auc_result = auc(fpr, tpr)
            logger.info(f"{model_name} AUC {data_type}: {roc_auc_result:.3f}")
            dresult = {"fpr": fpr, "tpr": tpr}
            dft = pd.DataFrame(dresult)
            dft["label"] = f"AUC {data_type}: {roc_auc_result:.3f}"
            dft["model_auc"] = f"{model_name} AUC: {roc_auc_result:.3f}"
            dft["auc"] = roc_auc_result
            return dft

        dftrain = create_results(X_train, Y_train, "Train")
        dftest = create_results(X_test, Y_test, "Test")
        result = pd.concat([dftrain, dftest])
    else:
        RMSE_TRAIN = np.sqrt(mean_squared_error(Y_train, Y_Pred_train))
        RMSE_TEST = np.sqrt(mean_squared_error(Y_test, Y_Pred_test))
        logger.info(f"{model_name} MEAN Train: {Y_train.mean()}")
        logger.info(f"{model_name} MEAN Test: {Y_test.mean()}")
        result = pd.DataFrame(
            {
                "Model": [model_name, model_name],
                "RMSE": [RMSE_TRAIN, RMSE_TEST],
                "label": ["Train", "Test"],
            }
        )
        logger.info(
            f"{model_name} RMSE Train: {RMSE_TRAIN} (Error Ratio: {RMSE_TRAIN/Y_train.mean()})"
        )
        logger.info(
            f"{model_name} RMSE Test: {RMSE_TEST} (Error Ratio: {RMSE_TEST/Y_test.mean()})"
        )
    return model, result


def plot_roc(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    title: str,
    xlabel: str,
    ylabel: str,
    save_img_path: str = None,
    figsize: tuple = (6, 4),
    legend_outside=False,
) -> None:
    """
    Plot ROC curves.

    Args:
        data (pd.DataFrame): Data containing ROC curve data points.
        x (str): Column name for the x-axis data.
        y (str): Column name for the y-axis data.
        hue (str): Column name to distinguish different curves.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        save_img_path (str, optional): File path to save the plot as an image. Defaults to None.
        figsize (tuple, optional): Figure size. Defaults to (6, 4).
    """
    fig, ax = plt.subplots(figsize=figsize)
    a = sns.lineplot(data=data, x=x, y=y, ax=ax, hue=hue)
    sns.lineplot(x=[0, 1], y=[0, 1], ax=ax, linestyle="--", color="gray")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    if legend_outside:
        sns.move_legend(a, "upper left", bbox_to_anchor=(1, 1))
    if save_img_path:
        fig.savefig(save_img_path, bbox_inches="tight")


def scale_data(X: pd.DataFrame, theScaler: object) -> pd.DataFrame:
    """Function to scale data.

    Args:
        X (pd.DataFrame): Data to be scaled
        theScaler (object): Fitted Scaler.

    Returns:
        pd.DataFrame: Scaled data
    """
    X_new = theScaler.transform(X)
    X_new = pd.DataFrame(X_new)
    X_new.columns = list(X.columns.values)
    return X_new


df = pd.read_csv(path + "/" + "HMEQ_Loss.csv")
TARGET_F = "TARGET_BAD_FLAG"
TARGET_A = "TARGET_LOSS_AMT"

# ------- Impute Missing Values -------
objcols = ["REASON", "JOB"]
targetcols = [TARGET_F, TARGET_A]
mask_numcols = ~df.columns.isin(objcols)
numcols = df.columns[mask_numcols]
for col in objcols:
    df.loc[df[col].isna(), col] = "Unknown"
for col in numcols:
    if col in targetcols or df[col].isna().sum() == 0:
        continue
    FLAG = f"M_{col}"
    IMPUTED = f"IMP_{col}"
    df[FLAG] = df[col].isna().astype(int)
    df[IMPUTED] = df[col].fillna(
        df.groupby(objcols, dropna=False)[col].transform("median")
    )
    df = df.drop(col, axis=1)

# ------- Create dummy variables -------
for col in objcols:
    thePrefix = "z_" + col
    y = pd.get_dummies(df[col], prefix=thePrefix, drop_first=False)
    y = y.drop(y.columns[-1], axis=1).astype(int)
    df = pd.concat([df, y], axis=1)
    df = df.drop(col, axis=1)

# -------- Setting Cap limit --------
cap = 25_000
cap_limit = df[TARGET_A] > cap
df.loc[cap_limit, TARGET_A] = cap

# -------- Split Data --------
X = df.copy()
X = X.drop([TARGET_F, TARGET_A], axis=1)
Y = df.loc[:, [TARGET_F, TARGET_A]].copy()
A_mask = Y[TARGET_A].notna()
XA = X[A_mask].copy()
YA = Y[A_mask].copy()
Y = Y.loc[:, TARGET_F]
YA = YA.loc[:, TARGET_A]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, train_size=0.8, test_size=0.2, random_state=random_state
)
XA_train, XA_test, YA_train, YA_test = train_test_split(
    XA, YA, train_size=0.8, test_size=0.2, random_state=random_state
)

# -------- Scale Data for TensorFlow --------
theScaler = MinMaxScaler()
theScaler.fit(X_train)
X_train_tf = scale_data(X=X_train, theScaler=theScaler)
X_test_tf = scale_data(X=X_test, theScaler=theScaler)
XA_train_tf = scale_data(X=XA_train, theScaler=theScaler)
XA_test_tf = scale_data(X=XA_test, theScaler=theScaler)

# -------- Variable selection --------
if variable_selection_model == "All":
    logger.info("Using all variables")
    logger.info(f"All Variables Count: {X_train.shape[1]}")
    logger.info(f"All Variables: {list(X_train.columns.values)}")
else:
    logger.info(f"Variables Selection enabled for {variable_selection_model}")
    importance_feat_class = get_important_features(
        model=model_dic[variable_selection_model]["Categorical"],
        X_train=X_train,
        Y_train=Y_train,
    )
    logger.info(
        f"Important Variables Count from {variable_selection_model} Categorical: {len(importance_feat_class)}"
    )
    logger.info(
        f"Important Variables from {variable_selection_model} Categorical: {importance_feat_class}"
    )
    importance_feat_regress = get_important_features(
        model=model_dic[variable_selection_model]["Regression"],
        X_train=XA_train,
        Y_train=YA_train,
    )
    logger.info(
        f"Important Variables from {variable_selection_model} Regression: {len(importance_feat_regress)}"
    )
    logger.info(
        f"Important Variables from {variable_selection_model} Regression: {importance_feat_regress}"
    )
    X_train = X_train.loc[:, importance_feat_class].copy()
    X_test = X_test.loc[:, importance_feat_class].copy()
    XA_train = XA_train.loc[:, importance_feat_regress].copy()
    XA_test = XA_test.loc[:, importance_feat_regress].copy()

# -------- Plotting Parameters and Labels for ROCAUC --------
xc = "fpr"
yc = "tpr"
huec = "label"
xlabelc = "False Positive Rate"
ylabelc = "True Positive Rate"

# -------- Model Iteration --------
roc_results_tree = {}
rmse_results_tree = {}
for idx, (name, model_types) in enumerate(model_dic.items()):
    for model_type, model in model_types.items():
        logger.info(f"{name} {model_type}")
        if model_type == "Categorical":
            isCat = True
            model_new, result = process_model(
                model=model,
                model_name=name,
                X_train=X_train,
                Y_train=Y_train,
                X_test=X_test,
                Y_test=Y_test,
                isCat=isCat,
            )
            titlec = f"{name} ROC Curve ({variable_selection_model} Parameters)"
            save_img_path = (
                img_path
                + f"ROC-{name}-var-{variable_selection_model}-rs{random_state}-md{max_depth}-ft{feature_thresh}.png"
            )
            plot_roc(
                data=result,
                x=xc,
                y=yc,
                hue=huec,
                title=titlec,
                xlabel=xlabelc,
                ylabel=ylabelc,
                save_img_path=save_img_path,
                figsize=figsize,
            )
            roc_results_tree[name] = result
        else:
            isCat = False
            model_new, result = process_model(
                model=model,
                model_name=name,
                X_train=XA_train,
                Y_train=YA_train,
                X_test=XA_test,
                Y_test=YA_test,
                isCat=isCat,
            )
            rmse_results_tree[name] = result
        logger.info("------------------------------------")

# -------- TensorFlow Iteration --------
tf_model_types = ["Categorical", "Regression"]
hidden_layers = [1, 2]
dropout_layers = [True, False]
roc_results_tf = {}
rmse_results_tf = {}
for act_name, activation in activations_names.items():
    for hlayers in hidden_layers:
        for dropout in dropout_layers:
            for model_type in tf_model_types:
                if model_type == "Categorical":
                    isCat = True
                    theShapeSize = X_train_tf.shape[1]
                    theLossMetric = tf.keras.losses.SparseCategoricalCrossentropy()
                    LAYER_OUTPUT = tf.keras.layers.Dense(
                        units=2, activation=tf.keras.activations.softmax
                    )
                    theEpochs = epochs_categorical
                else:
                    isCat = False
                    theShapeSize = XA_train_tf.shape[1]
                    theLossMetric = tf.keras.losses.MeanSquaredError()
                    LAYER_OUTPUT = tf.keras.layers.Dense(
                        units=1, activation=tf.keras.activations.linear
                    )
                    theEpochs = epochs_regression
                name = f"TF-{act_name}-hl-{hlayers}-do-{dropout}-ep-{theEpochs}"
                theOptimizer = tf.keras.optimizers.Adam()
                theUnits = get_units(theShapeSize)
                LAYER_01 = tf.keras.layers.Dense(
                    units=theUnits, activation=activation, input_dim=theShapeSize
                )
                LAYER_DROP = tf.keras.layers.Dropout(drop_out_ratio)
                LAYER_02 = tf.keras.layers.Dense(units=theUnits, activation=activation)

                TFM = tf.keras.Sequential()
                TFM.add(LAYER_01)
                if dropout:
                    TFM.add(LAYER_DROP)
                if hlayers == 2:
                    TFM.add(LAYER_02)
                    if dropout:
                        TFM.add(LAYER_DROP)
                TFM.add(LAYER_OUTPUT)
                TFM.compile(loss=theLossMetric, optimizer=theOptimizer)
                if model_type == "Categorical":
                    model_new, result = process_model(
                        model=TFM,
                        model_name=name,
                        X_train=X_train_tf,
                        Y_train=Y_train,
                        X_test=X_test_tf,
                        Y_test=Y_test,
                        isCat=isCat,
                        theEpochs=theEpochs,
                    )
                    titlec = (
                        f"{name}: ROC Curve ({variable_selection_model} Parameters)"
                    )
                    save_img_path = (
                        img_path
                        + f"ROC-{name}-var-{variable_selection_model}-rs{random_state}-md{max_depth}-ft{feature_thresh}.png"
                    )
                    plot_roc(
                        data=result,
                        x=xc,
                        y=yc,
                        hue=huec,
                        title=titlec,
                        xlabel=xlabelc,
                        ylabel=ylabelc,
                        save_img_path=save_img_path,
                        figsize=figsize,
                    )
                    roc_results_tf[f"{titlec}"] = result
                else:
                    model_new, result = process_model(
                        model=TFM,
                        model_name=name,
                        X_train=XA_train_tf,
                        Y_train=YA_train,
                        X_test=XA_test_tf,
                        Y_test=YA_test,
                        isCat=isCat,
                        theEpochs=theEpochs,
                    )
                    rmse_results_tf[name] = result
                logger.info("------------------------------------")

# -------- Comparing all results and saving --------
roc_df_tf = pd.concat([d for d in roc_results_tf.values()])
roc_df_tf = roc_df_tf[roc_df_tf["label"].str.contains("Test")].reset_index(drop=True)
rmse_df_tf = pd.concat([d for d in rmse_results_tf.values()])
rmse_df_tf = rmse_df_tf[rmse_df_tf["label"].str.contains("Test")].reset_index(drop=True)
save_img_path_tf = (
    img_path
    + f"ROC-TF-Comparison--ep{theEpochs}-var-{variable_selection_model}-rs{random_state}-md{max_depth}-ft{feature_thresh}.png"
)
title_comapre_tf = (
    f"ROC Curve Tensor Flow Comparison ({variable_selection_model} Parameters)"
)
plot_roc(
    data=roc_df_tf,
    x=xc,
    y=yc,
    hue="model_auc",
    title=title_comapre_tf,
    xlabel=xlabelc,
    ylabel=ylabelc,
    save_img_path=save_img_path_tf,
    figsize=figsize,
    legend_outside=True,
)

roc_df_tree = pd.concat([d for d in roc_results_tree.values()])
roc_df_tree = roc_df_tree[roc_df_tree["label"].str.contains("Test")].reset_index(
    drop=True
)
rmse_df_tree = pd.concat([d for d in rmse_results_tree.values()])
rmse_df_tree = rmse_df_tree[rmse_df_tree["label"].str.contains("Test")].reset_index(
    drop=True
)
roc_df_tf_best = roc_df_tf[
    (roc_df_tf["label"].str.contains("Test"))
    & (roc_df_tf["auc"] == roc_df_tf["auc"].max())
].reset_index(drop=True)

roc_df_comparison = pd.concat([roc_df_tree, roc_df_tf_best]).reset_index(drop=True)
rmse_df_comparison = pd.concat([rmse_df_tree, rmse_df_tf]).reset_index(drop=True)

save_img_path_comparison = (
    img_path
    + f"ROC-Comparison--ep{theEpochs}-var-{variable_selection_model}-rs{random_state}-md{max_depth}-ft{feature_thresh}.png"
)
title_comapre_all = f"ROC Curve Comparison ({variable_selection_model} Parameters)"
plot_roc(
    data=roc_df_comparison,
    x=xc,
    y=yc,
    hue="model_auc",
    title=title_comapre_all,
    xlabel=xlabelc,
    ylabel=ylabelc,
    save_img_path=save_img_path_comparison,
    figsize=figsize,
)
rmse_df_comparison = rmse_df_comparison.sort_values("RMSE").reset_index(drop=True)
save_df_path = (
    log_path
    + f"RMSE-Comparison--ep{theEpochs}-var-{variable_selection_model}-rs{random_state}-md{max_depth}-ft{feature_thresh}.csv"
)
rmse_df_comparison.to_csv(save_df_path, index=False)
