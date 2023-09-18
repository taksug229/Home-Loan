import os
import sys
import numpy as np
import pandas as pd
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, roc_curve, auc, mean_squared_error

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.config import (
    TARGET_A,
    targetcols,
    objcols,
    numcols,
    max_depth,
    grid_search_dic,
)


def data_imputation(
    df: pd.DataFrame,
    objcols: List[str] = objcols,
    numcols: List[str] = numcols,
    targetcols: List[str] = targetcols,
) -> pd.DataFrame:
    """
    Impute missing values in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        objcols (List[str], optional): A list of categorical column names. Defaults to setting in config.
        numcols (List[str], optional): A list of numerical column names. Defaults to setting in config.
        targetcols (List[str], optional): A list of target column names. Defaults to setting in config.
    Returns:
        pd.DataFrame: The DataFrame with missing values imputed.
    """
    data = df.copy()
    for col in objcols:
        data.loc[data[col].isna(), col] = "Unknown"
    for col in numcols:
        if col in targetcols or data[col].isna().sum() == 0:
            continue
        FLAG = f"M_{col}"
        IMPUTED = f"IMP_{col}"
        data[FLAG] = data[col].isna().astype(int)
        data[IMPUTED] = data[col].fillna(
            data.groupby(objcols, dropna=False)[col].transform("median")
        )
        data = data.drop(col, axis=1)
    return data


def data_dummy(
    df: pd.DataFrame,
    objcols: List[str] = objcols,
) -> pd.DataFrame:
    """
    Create dummy variables for categorical columns in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        objcols (List[str], optional): A list of categorical column names. Defaults to setting in config.

    Returns:
        pd.DataFrame: The DataFrame with dummy variables created.
    """
    data = df.copy()
    for col in objcols:
        thePrefix = "z_" + col
        y = pd.get_dummies(data[col], prefix=thePrefix, drop_first=False)
        y = y.drop(y.columns[-1], axis=1).astype(int)
        data = pd.concat([data, y], axis=1)
        data = data.drop(col, axis=1)
    return data


def data_cap(
    df: pd.DataFrame, TARGET_A: str = TARGET_A, cap: int = 25_000
) -> pd.DataFrame:
    """
    Cap values in a specific column of a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        TARGET_A (str, optional): The name of the target column to be capped. Defaults to setting in config.
        cap (int, optional): The value to which the target column values will be capped. Defaults to setting in config.

    Returns:
        pd.DataFrame: The DataFrame with capped values.
    """
    data = df.copy()
    cap_limit = data[TARGET_A] > cap
    data.loc[cap_limit, TARGET_A] = cap
    return data


def preprocess_data_with_log(
    df: pd.DataFrame,
    logger: logging.RootLogger,
    objcols: List[str] = objcols,
    numcols: List[str] = numcols,
    targetcols: List[str] = targetcols,
    TARGET_A: str = TARGET_A,
) -> pd.DataFrame:
    """
    Preprocess data with logging.

    Args:
        df (pd.DataFrame): The input DataFrame.
        logger (logging.RootLogger): The logger object for logging.
        objcols (List[str], optional): A list of categorical column names. Defaults to setting in config.
        numcols (List[str], optional): A list of numerical column names. Defaults to setting in config.
        targetcols (List[str], optional): A list of target column names. Defaults to setting in config.
        TARGET_F (str, optional): The name of the binary target column. Defaults to setting in config.
        TARGET_A (str, optional): The name of the numeric target column. Defaults to setting in config.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    data = df.copy()
    logger.info("Total NaN values in each row.")
    logger.info(
        "NOTE: I am not imputing TARGET_LOSS_AMT because this is only NaN when the person did not default."
    )
    nasumb = data.isna().sum().to_string()
    logger.info(f"\n{nasumb}")
    logger.info(
        "Median of each numerical column grouped by job and reason of loan. I will impute NaNs with these values."
    )
    gbimp = data.groupby(objcols, dropna=False)[numcols].median().to_string()
    logger.info(f"\n{gbimp}")

    # ------- Impute Missing Values -------
    logger.info("Impute missing values START.")
    logger.info("Impute missing categorical values to 'Unknwon'.")
    logger.info("Add new column for flag and imputed numbers.")
    logger.info("The NaNs will be imputed by the median of per Job and Reason.")
    logger.info("The original imputed columns will be deleted.")
    data = data_imputation(
        df=data, objcols=objcols, numcols=numcols, targetcols=targetcols
    )
    logger.info("Impute missing values END.")
    logger.info("First 5 observations of the imputed data")
    head = df.head().T.to_string()
    logger.info(f"\n{head}")
    logger.info(
        "Confirming there are no NaNs (except the intentionally left TARGET_LOSS_AMT.)"
    )
    nasuma = data.isna().sum().to_string()
    logger.info(f"\n{nasuma}")

    # ------- Create dummy variables -------
    logger.info("Creating dummy variables for categorical columns")
    data = data_dummy(df=data, objcols=objcols)
    logger.info("Completed creating dummy variables")
    logger.info("First 5 observations of the data with dummy variables")
    headdum = data.head().to_string()
    logger.info(f"\n{headdum}")

    # ------- Cap loss -------
    data = data_cap(df=data, TARGET_A=TARGET_A)
    return data

    # logger.info("Saving data")
    # save_csv_path = path + "/" + "NEW_HMEQ_LOSS.csv"
    # df.to_csv(save_csv_path, index=False)


def save_graph(
    df: pd.DataFrame, x: str, y: str, hue: str, title: str, save_img_path: str
) -> None:
    """Function to save graph

    Args:
        df (pd.DataFrame): Dataframe used for plotting.
        x (str): x variable column for barplot.
        y (str): y variable column for barplot.
        hue (str): hue variable column for barplot.
        title (str): Title for the graph.
        save_img_path (str): Image save path.
    Returns:
        None
    """
    fig, ax = plt.subplots()
    a = sns.barplot(ax=ax, x=x, y=y, hue=hue, data=df)
    a.set(title=title)
    a.grid(axis="y")
    fig.savefig(save_img_path, bbox_inches="tight")


def process_model(
    model,
    model_name,
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    Y_test: pd.DataFrame,
    TARGET: str,
    isCat: bool,
    logger: logging.RootLogger,
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
        logger (logging.RootLogger): The logger to write logs on.
        UseGridSearch (bool, optional): If True, Grid Search parameters will be used. Defaults to False.
        cv (int, optional): Number of cross-validation folds. Defaults to 5.

    Returns:
        tuple: A tuple containing the trained model, result DataFrame with evaluation metrics,
            cross-validation scores, and DataFrame containing feature importances.
    """
    if UseGridSearch and isinstance(grid_search_dic, dict):
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
        fig.savefig(save_img_path, bbox_inches="tight")


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
        fig.savefig(save_img_path, bbox_inches="tight")
