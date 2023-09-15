import os
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, roc_curve, auc, mean_squared_error
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn import tree
import graphviz

# -------- Settings --------
random_state = 1  # Data split random_state
cv = 5  # Cross Validation K Folds
figsize = (6, 4)  # Figure size for plots

# -------- Default Parameters --------
max_depth = 3  # Depth size for tree

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

# -------- Configure output path and logging --------
path = os.path.dirname(os.path.realpath(__file__))
img_path = path + "/img/"
viz_path = path + "/viz/"
os.makedirs(img_path, exist_ok=True)
os.makedirs(viz_path, exist_ok=True)

current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
log_file_path = path + "/" + f"analysis-log-{title_params}-{formatted_datetime}.log"
logging.basicConfig(
    level=logging.INFO,
    filename=log_file_path,
    filemode="w",
    format="%(asctime)s - %(levelname)s: %(message)s",
)
logger = logging.getLogger()


# ------- Functions ----------
def process_model(
    model,
    model_name,
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    Y_test: pd.DataFrame,
    TARGET: str,
    isCat: bool,
    UseGridSearch: bool = False,
    cv: int = 5,
) -> tuple:
    """
    Train and evaluate a machine learning model with optional Grid Search.

    Args:
        model (object): The machine learning model to be trained and evaluated.
        model_name (str): Name of the model.
        X_train (pd.DataFrame): Features of the training data.
        Y_train (pd.DataFrame): Target variables of the training data.
        X_test (pd.DataFrame): Features of the test data.
        Y_test (pd.DataFrame): Target variables of the test data.
        TARGET (str): Name of the target variable.
        isCat (bool): True if the problem is a classification task, False if regression.
        UseGridSearch (bool, optional): If True, Grid Search parameters will be used.
            Defaults to False.
        cv (int, optional): Number of cross-validation folds. Defaults to 5.

    Returns:
        tuple: A tuple containing the trained model, result DataFrame with evaluation metrics,
            cross-validation scores, and DataFrame containing feature importances.
    """
    if UseGridSearch:
        global logger, grid_search_dic
        logger.info(f"Using Grid Search for {model_name}.")
        grid_search = GridSearchCV(model, grid_search_dic[model_name], cv=cv)
        grid_search.fit(X_train, Y_train[TARGET])
        best_params = grid_search.best_params_
        model.set_params(**best_params)
        if isCat:
            logger.info(f"{model_name} DEFAULT RISK best parameters = {best_params}")
        else:
            logger.info(f"{model_name} LOSS AMOUNT best parameters = {best_params}")
    else:
        logger.info(f"Using Default Parameters for {model_name}.")
        model.set_params(**{"max_depth": max_depth})
    model_ = model.fit(X_train, Y_train[TARGET])
    Y_Pred_train = model_.predict(X_train)
    Y_Pred_test = model_.predict(X_test)
    if isCat:
        train_accuracy = accuracy_score(Y_train[TARGET], Y_Pred_train)
        test_accuracy = accuracy_score(Y_test[TARGET], Y_Pred_test)
        logger.info(f"{model_name} Accuracy Train: {train_accuracy:.4f}")
        logger.info(f"{model_name} Accuracy Test:: {test_accuracy:.4f}")

        def create_results(X, Y, data_type):
            probs = model_.predict_proba(X)
            p = probs[:, 1]
            fpr, tpr, _ = roc_curve(Y[TARGET], p)
            roc_auc_result = auc(fpr, tpr)
            logger.info(f"{model_name} AUC {data_type}: {roc_auc_result:.2f}")
            dresult = {"fpr": fpr, "tpr": tpr}
            dft = pd.DataFrame(dresult)
            dft["label"] = f"AUC {data_type}: {roc_auc_result:.2f}"
            dft["model_auc"] = f"{model_name} AUC: {roc_auc_result:.2f}"
            return dft

        dftrain = create_results(X_train, Y_train, "Train")
        dftest = create_results(X_test, Y_test, "Test")
        result = pd.concat([dftrain, dftest])
        cv_scores = cross_val_score(
            model_, X_train, Y_train[TARGET], scoring="roc_auc", cv=cv
        )
        logger.info(
            f"{model_name} AUC Train Cross Validation Average: {np.mean(cv_scores):.2f}"
        )
    else:
        RMSE_TRAIN = np.sqrt(mean_squared_error(Y_train[TARGET], Y_Pred_train))
        RMSE_TEST = np.sqrt(mean_squared_error(Y_test[TARGET], Y_Pred_test))
        logger.info(f"{model_name} MEAN Train: {Y_train[TARGET].mean()}")
        logger.info(f"{model_name} MEAN Test: {Y_test[TARGET].mean()}")
        result = pd.DataFrame(
            {"RMSE": [RMSE_TRAIN, RMSE_TEST], "label": [f"TRAIN", f"TEST"]}
        )
        cv_scores = cross_val_score(
            model_,
            X_train,
            Y_train[TARGET],
            scoring="neg_root_mean_squared_error",
            cv=cv,
        )
        cv_scores *= -1
        logger.info(f"{model_name} RMSE Train: {RMSE_TRAIN}")
        logger.info(f"{model_name} RMSE Test: {RMSE_TEST}")
        logger.info(
            f"{model_name} RMSE Train Cross Validation Average: {np.mean(cv_scores):.2f}"
        )

    importance_df = pd.DataFrame(
        {"Feature": X_train.columns, "Importance": model_.feature_importances_}
    )
    importance_df = importance_df.sort_values(by="Importance", ascending=False)
    importance_df = importance_df[importance_df["Importance"] > 0]
    return model_, result, cv_scores, importance_df


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


def plot_importance(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color: str = sns.color_palette("tab10")[0],
    save_img_path: str = None,
    figsize: tuple = (6, 4),
) -> None:
    """
    Plot feature importances.

    Args:
        data (pd.DataFrame): Data containing feature importances.
        x (str): Column name for the x-axis data.
        y (str): Column name for the y-axis data.
        title (str): Title of the plot.
        color (str, optional): Color for the bars. Defaults to sns.color_palette("tab10")[0].
        save_img_path (str, optional): File path to save the plot as an image. Defaults to None.
        figsize (tuple, optional): Figure size. Defaults to (6, 4).
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=x, y=y, data=data, ax=ax, color=color)
    ax.set(title=title)
    if save_img_path:
        fig.savefig(save_img_path)


# -------- Reading data, prerpocessing, splitting data. ----------

df = pd.read_csv(path + "/" + "NEW_HMEQ_LOSS.csv")
TARGET_F = "TARGET_BAD_FLAG"
TARGET_A = "TARGET_LOSS_AMT"
cap = 25_000
cap_limit = df[TARGET_A] > cap
df.loc[cap_limit, TARGET_A] = cap

logger.info("LOSS AMOUNT statistics")
logger.info(f"\n{df[TARGET_A].describe().to_string()}")

logger.info("Data Split")
X = df.copy()
X = X.drop([TARGET_F, TARGET_A], axis=1)
Y = df.loc[:, [TARGET_F, TARGET_A]].copy()
A_mask = Y[TARGET_A].notna()
XA = X[A_mask].copy()
YA = Y[A_mask].copy()

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, train_size=0.8, test_size=0.2, random_state=random_state, stratify=Y[TARGET_F]
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

# ----------- Models ----------
loop_dic = {
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
            model_new, result, cv_scores, important_vars = process_model(
                model=model,
                model_name=name,
                X_train=X_train,
                Y_train=Y_train,
                X_test=X_test,
                Y_test=Y_test,
                TARGET=TARGET_F,
                isCat=isCat,
                UseGridSearch=UseGridSearch,
                cv=cv,
            )
            titlec = f"{name} ROC Curve ({title_params})"
            save_img_path = img_path + f"ROC-{name}-{title_params}-{random_state}.png"
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
            model_new, result, cv_scores, important_vars = process_model(
                model=model,
                model_name=name,
                X_train=XA_train,
                Y_train=YA_train,
                X_test=XA_test,
                Y_test=YA_test,
                TARGET=TARGET_A,
                isCat=isCat,
                UseGridSearch=UseGridSearch,
                cv=cv,
            )
        logger.info(f'Important Variables\n{important_vars.query("Importance > 0")}')

        title = f"{name} {model_type}: Feature Importance ({title_params})"
        color = sns.color_palette("tab10")[idx]
        save_img_path = (
            img_path
            + f"feature_importance-{name}-{model_type}-{title_params}-{random_state}.png"
        )
        plot_importance(
            data=important_vars,
            x=x,
            y=y,
            title=title,
            color=color,
            save_img_path=save_img_path,
            figsize=figsize,
        )

        if name == "Decision Tree":
            feature_cols = list(X.columns.values)
            viz_file_name = (
                f"Decition_Tree_{model_type}_{title_params}_viz-{random_state}"
            )
            graphviz_data = tree.export_graphviz(
                model_new,
                out_file=None,
                filled=True,
                rounded=True,
                feature_names=feature_cols,
                class_names=["Good", "Bad"],
                impurity=False,
                precision=0,
            )
            graph = graphviz.Source(graphviz_data, format="png")
            graph.render(filename=viz_file_name, directory=viz_path, cleanup=True)
            logger.info(f"Saved viz file to: {viz_path + '/' + viz_file_name}")

# ---------- Compare ROC Curves ----------
roc_df = pd.concat([d for d in roc_results.values()])
roc_df = roc_df[roc_df["label"].str.contains("Test")].reset_index(drop=True)

save_img_path = img_path + f"ROC-Comparison-{title_params}-{random_state}.png"
plot_roc(
    data=roc_df,
    x=xc,
    y=yc,
    hue="model_auc",
    title=f"ROC Curve Comparison with Test Set ({title_params})",
    xlabel=xlabelc,
    ylabel=ylabelc,
    save_img_path=save_img_path,
    figsize=figsize,
)
print("Completed!")
print(f"Saved images to {img_path}")
print(f"Saved graphviz images to {viz_path}")
print(f"Saved logs for analysis to {log_file_path}")
