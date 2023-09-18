import os
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.base import BaseEstimator
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_curve, auc, mean_squared_error
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn import tree

# -------- Settings --------
random_state = 1  # Data split random_state
cv = 5  # Cross Validation K Folds
figsize = (6, 4)  # Figure size for plots

# -------- Default Parameters --------
max_depth = 3  # Max depth size for tree
feature_thresh = 0.05  # Feature importance threshold for tree models
max_step_wise_vars = 10  # Max step wise variables

# -------- Configure output path and logging --------
path = os.path.dirname(os.path.realpath(__file__))
img_path = path + "/img/"
os.makedirs(img_path, exist_ok=True)

current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
log_file_path = (
    path
    + "/"
    + f"analysis-log--rs{random_state}-cv{cv}-md{max_depth}-ft{feature_thresh}-mv{max_step_wise_vars}-{formatted_datetime}.log"
)
logging.basicConfig(
    level=logging.INFO,
    filename=log_file_path,
    filemode="w",
    format="%(asctime)s - %(levelname)s: %(message)s",
)
logger = logging.getLogger()


# ------- Functions ----------
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
    return importance_feat


@ignore_warnings(category=ConvergenceWarning)
def get_step_wise_features(
    model: BaseEstimator,
    model_type: str,
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
    global logger, random_state
    model.fit(X_train, Y_train)
    fig, ax = plot_sfs(metric_dict=model.get_metric_dict(), kind=None, figsize=figsize)
    ax.set_title(f"{model_type}: Sequential Forward Selection")
    save_img_path = (
        img_path
        + f"{model_type}-Stepwise-rs{random_state}-cv{cv}-md{max_depth}-ft{feature_thresh}-mv{max_step_wise_vars}.png"
    )
    fig.savefig(save_img_path)
    dfm = pd.DataFrame.from_dict(model.get_metric_dict()).T
    dfm = dfm[["feature_names", "avg_score"]]
    dfm.avg_score = dfm.avg_score.astype(float)
    maxIndex = dfm.avg_score.argmax()
    logger.info(f"{model_type} best score: {dfm['avg_score'].iloc[ maxIndex ]} ")
    stepVars = dfm.iloc[maxIndex,]
    stepVars = list(stepVars.feature_names)
    return stepVars


@ignore_warnings(category=ConvergenceWarning)
def process_model(
    model: BaseEstimator,
    model_name: str,
    X_train: pd.DataFrame,
    Y_train: pd.Series,
    X_test: pd.DataFrame,
    Y_test: pd.Series,
    isCat: bool,
    cv: int = 5,
) -> tuple:
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
        cv (int, optional): Number of cross-validation folds. Defaults to 5.

    Returns:
        tuple: A tuple containing the trained model, result DataFrame with evaluation metrics,
            cross-validation scores, and DataFrame containing feature importances.
    """

    logger.info(f"Using Features from {model_name}.")
    model_ = model.fit(X_train, Y_train)
    Y_Pred_train = model_.predict(X_train)
    Y_Pred_test = model_.predict(X_test)
    if isCat:
        train_accuracy = accuracy_score(Y_train, Y_Pred_train)
        test_accuracy = accuracy_score(Y_test, Y_Pred_test)
        logger.info(f"{model_name} variables Accuracy Train: {train_accuracy:.4f}")
        logger.info(f"{model_name} variables  Accuracy Test:: {test_accuracy:.4f}")

        def create_results(X, Y, data_type):
            probs = model_.predict_proba(X)
            p = probs[:, 1]
            fpr, tpr, _ = roc_curve(Y, p)
            roc_auc_result = auc(fpr, tpr)
            logger.info(f"{model_name} variables AUC {data_type}: {roc_auc_result:.2f}")
            dresult = {"fpr": fpr, "tpr": tpr}
            dft = pd.DataFrame(dresult)
            dft["label"] = f"AUC {data_type}: {roc_auc_result:.2f}"
            dft["model_auc"] = f"{model_name} variables AUC: {roc_auc_result:.2f}"
            return dft

        dftrain = create_results(X_train, Y_train, "Train")
        dftest = create_results(X_test, Y_test, "Test")
        result = pd.concat([dftrain, dftest])
        cv_scores = cross_val_score(model_, X_train, Y_train, scoring="roc_auc", cv=cv)
        logger.info(
            f"{model_name} AUC Train Cross Validation Average: {np.mean(cv_scores):.2f}"
        )
    else:
        RMSE_TRAIN = np.sqrt(mean_squared_error(Y_train, Y_Pred_train))
        RMSE_TEST = np.sqrt(mean_squared_error(Y_test, Y_Pred_test))
        logger.info(f"{model_name} MEAN Train: {Y_train.mean()}")
        logger.info(f"{model_name} MEAN Test: {Y_test.mean()}")
        result = pd.DataFrame(
            {"RMSE": [RMSE_TRAIN, RMSE_TEST], "label": [f"TRAIN", f"TEST"]}
        )
        cv_scores = cross_val_score(
            model_,
            X_train,
            Y_train,
            scoring="neg_root_mean_squared_error",
            cv=cv,
        )
        cv_scores *= -1
        logger.info(
            f"{model_name} RMSE Train: {RMSE_TRAIN} (Error Ratio: {RMSE_TRAIN/Y_train.mean()})"
        )
        logger.info(
            f"{model_name} RMSE Test: {RMSE_TEST} (Error Ratio: {RMSE_TEST/Y_test.mean()})"
        )
        logger.info(
            f"{model_name} RMSE Train Cross Validation Average: {np.mean(cv_scores):.2f} (Error Ratio: {np.mean(cv_scores)/Y_train.mean()})"
        )

    return model_, result, cv_scores


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
    sns.lineplot(data=data, x=x, y=y, ax=ax, hue=hue)
    sns.lineplot(x=[0, 1], y=[0, 1], ax=ax, linestyle="--", color="gray")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    if save_img_path:
        fig.savefig(save_img_path)


def getCoef(MODEL: BaseEstimator, TRAIN_DATA: pd.DataFrame, isCat: bool) -> None:
    """
    Logs coefficients for the variables.

    Args:
        MODEL (BaseEstimator): The machine learning model.
        TRAIN_DATA (pd.DataFrame): The training data.
        isCat (bool): True if the problem is a classification task, False if regression.
    """
    global logger
    varNames = list(TRAIN_DATA.columns.values)
    coef_dict = {}
    if isCat:
        coef_dict["INTERCEPT"] = MODEL.intercept_[0]
        for coef, feat in zip(MODEL.coef_[0], varNames):
            coef_dict[feat] = coef
    else:
        coef_dict["INTERCEPT"] = MODEL.intercept_
        for coef, feat in zip(MODEL.coef_, varNames):
            coef_dict[feat] = coef
    logger.info(f"Total Variables: {len( varNames )} ")
    for i in coef_dict:
        logger.info(f"{i} = {coef_dict[i]}")


# -------- Reading data, prerpocessing, splitting data. ----------

df = pd.read_csv(path + "/" + "NEW_HMEQ_LOSS.csv")
TARGET_F = "TARGET_BAD_FLAG"
TARGET_A = "TARGET_LOSS_AMT"
# Setting Cap limit
cap = 25_000
cap_limit = df[TARGET_A] > cap
df.loc[cap_limit, TARGET_A] = cap

logger.info("LOSS AMOUNT statistics")
logger.info(f"\n{df[TARGET_A].describe().to_string()}")
logger.info("------------------------------------")
logger.info("Data Split")
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

logger.info("FLAG DATA")
logger.info(f"TRAINING = {X_train.shape}")
logger.info(f"TEST = {X_test.shape}")

logger.info("LOSS AMOUNT DATA")
logger.info(f"TRAINING = {YA_train.shape}")
logger.info(f"TEST = {YA_test.shape}")
logger.info(f"\n{YA_train.describe().to_string()}")
logger.info(f"\n{YA_test.describe().to_string()}")
logger.info("------------------------------------")
# ----------- Models ----------
loop_dic = {
    "All": {
        "Categorical": None,
        "Regression": None,
    },
    "Decision Tree": {
        "Categorical": tree.DecisionTreeClassifier(),
        "Regression": tree.DecisionTreeRegressor(),
    },
    "Random Forest": {
        "Categorical": RandomForestClassifier(),
        "Regression": RandomForestRegressor(),
    },
    "Gradient Boosting": {
        "Categorical": GradientBoostingClassifier(),
        "Regression": GradientBoostingRegressor(),
    },
    "Step Wise": {
        "Categorical": SFS(
            LogisticRegression(),
            k_features=(1, max_step_wise_vars),
            forward=True,
            floating=False,
            verbose=2,
            scoring="roc_auc",
            cv=cv,
        ),
        "Regression": SFS(
            LinearRegression(),
            k_features=(1, max_step_wise_vars),
            forward=True,
            floating=False,
            verbose=2,
            scoring="neg_root_mean_squared_error",
            cv=cv,
        ),
    },
}

# Plotting Parameters and Labels for ROCAUC
xc = "fpr"
yc = "tpr"
huec = "label"
xlabelc = "False Positive Rate"
ylabelc = "True Positive Rate"

# Plotting Parameters and Labels for Feature Importance
x = "Importance"
y = "Feature"

# ----------- Looping through each model, logging results, saving outputs ----------
roc_results = {}
for idx, (name, model_types) in enumerate(loop_dic.items()):
    for model_type, model in model_types.items():
        logger.info(f"{name} {model_type}")
        if model_type == "Categorical":
            isCat = True
            if name in ["Decision Tree", "Random Forest", "Gradient Boosting"]:
                importance_feat = get_important_features(
                    model=model,
                    X_train=X_train,
                    Y_train=Y_train,
                )
                logger.info(f"Important Variables from {name}: {importance_feat}")
                X_train_ = X_train.loc[:, importance_feat].copy()
                X_test_ = X_test.loc[:, importance_feat].copy()
            elif name == "Step Wise":
                importance_feat = get_step_wise_features(
                    model=model,
                    model_type=model_type,
                    X_train=X_train,
                    Y_train=Y_train,
                )
                logger.info(f"Important Variables from {name}: {importance_feat}")
                X_train_ = X_train.loc[:, importance_feat].copy()
                X_test_ = X_test.loc[:, importance_feat].copy()
            else:
                X_train_ = X_train.copy()
                X_test_ = X_test.copy()

            model_new, result, cv_scores = process_model(
                model=LogisticRegression(),
                model_name=name,
                X_train=X_train_,
                Y_train=Y_train,
                X_test=X_test_,
                Y_test=Y_test,
                isCat=isCat,
                cv=cv,
            )
            getCoef(model_new, X_train_, isCat)
            titlec = f"Logistic Regression ROC Curve ({name} variables)"
            save_img_path = (
                img_path
                + f"ROC-{name}-features--rs{random_state}-cv{cv}-md{max_depth}-ft{feature_thresh}-mv{max_step_wise_vars}.png"
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
            roc_results[name] = result
        else:
            isCat = False
            if name in ["Decision Tree", "Random Forest", "Gradient Boosting"]:
                importance_feat = get_important_features(
                    model=model,
                    X_train=XA_train,
                    Y_train=YA_train,
                )
                logger.info(f"Important Variables from {name}: {importance_feat}")
                XA_train_ = XA_train.loc[:, importance_feat].copy()
                XA_test_ = XA_test.loc[:, importance_feat].copy()
            elif name == "Step Wise":
                importance_feat = get_step_wise_features(
                    model=model,
                    model_type=model_type,
                    X_train=XA_train,
                    Y_train=YA_train,
                )
                logger.info(f"Important Variables from {name}: {importance_feat}")
                XA_train_ = XA_train.loc[:, importance_feat].copy()
                XA_test_ = XA_test.loc[:, importance_feat].copy()
            else:
                XA_train_ = XA_train.copy()
                XA_test_ = XA_test.copy()

            model_new, result, cv_scores = process_model(
                model=LinearRegression(),
                model_name=name,
                X_train=XA_train_,
                Y_train=YA_train,
                X_test=XA_test_,
                Y_test=YA_test,
                isCat=isCat,
                cv=cv,
            )
            getCoef(model_new, XA_train_, isCat)
        logger.info("------------------------------------")
# ---------- Compare ROC Curves ----------
roc_df = pd.concat([d for d in roc_results.values()])
roc_df = roc_df[roc_df["label"].str.contains("Test")].reset_index(drop=True)

save_img_path = (
    img_path
    + f"ROC-Comparison--rs{random_state}-cv{cv}-md{max_depth}-ft{feature_thresh}-mv{max_step_wise_vars}.png"
)
plot_roc(
    data=roc_df,
    x=xc,
    y=yc,
    hue="model_auc",
    title="ROC Curve Comparison with Test Set",
    xlabel=xlabelc,
    ylabel=ylabelc,
    save_img_path=save_img_path,
    figsize=figsize,
)
print("\nCompleted!")
print(f"Saved images to {img_path}")
print(f"Saved logs for analysis to {log_file_path}")
