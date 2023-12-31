import os
import sys
import numpy as np
import pandas as pd
from typing import List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn import tree
from sklearn.base import BaseEstimator
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, mean_squared_error

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.config import (
    TARGET_F,
    TARGET_A,
    targetcols,
    objcols,
    numcols,
    figsize,
    random_state,
    cv,
    max_depth,
    feature_thresh,
    max_step_wise_vars,
    verbose,
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
    logger.info("LOSS AMOUNT statistics BEFORE capping")
    logger.info(f"\n{data[TARGET_A].describe().to_string()}")
    data = data_cap(df=data, TARGET_A=TARGET_A)

    logger.info("LOSS AMOUNT statistics AFTER capping")
    logger.info(f"\n{data[TARGET_A].describe().to_string()}")
    return data


def split_data(
    df: pd.DataFrame,
    logger: logging.RootLogger,
    TARGET_F: str = TARGET_F,
    TARGET_A: str = TARGET_A,
    random_state: int = random_state,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Split the input DataFrame into training and test sets for two target columns.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        logger (logging.RootLogger): The logger object for logging information.
        TARGET_F (str, optional): The name of the first target column. Defaults to TARGET_F from config file.
        TARGET_A (str, optional): The name of the second target column. Defaults to TARGET_A from config file.
        random_state (int, optional): Random seed for reproducibility. random_state from config file.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - X_train (pd.DataFrame): Features for training.
            - X_test (pd.DataFrame): Features for testing.
            - Y_train (pd.DataFrame): Both target columns for training.
            - Y_test (pd.DataFrame): Both target columns for testing.
            - XA_train (pd.DataFrame): Features for training (filtered by the presence of TARGET_A).
            - XA_test (pd.DataFrame): Features for testing (filtered by the presence of TARGET_A).
            - YA_train (pd.DataFrame): Both target columns for training (filtered by the presence of TARGET_A).
            - YA_test (pd.DataFrame): Both target columns for testing (filtered by the presence of TARGET_A).
    """
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
        X,
        Y,
        train_size=0.8,
        test_size=0.2,
        random_state=random_state,
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
    return X_train, X_test, Y_train, Y_test, XA_train, XA_test, YA_train, YA_test


def process_model(
    model: BaseEstimator,
    model_name: str,
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    Y_test: pd.DataFrame,
    isCat: bool,
    logger: logging.RootLogger,
    UseGridSearch: bool = False,
    cv: int = cv,
) -> tuple:
    """
    Train and evaluate a machine learning model with optional Grid Search.

    Args:
        model (BaseEstimator): The machine learning model to be trained and evaluated.
        model_name (str): Name of the model.
        X_train (pd.DataFrame): Features of the training data.
        Y_train (pd.DataFrame): Target variables of the training data.
        X_test (pd.DataFrame): Features of the test data.
        Y_test (pd.DataFrame): Target variables of the test data.
        isCat (bool): True if the problem is a classification task, False if regression.
        logger (logging.RootLogger): The logger to write logs on.
        UseGridSearch (bool, optional): If True, Grid Search parameters will be used. Defaults to False.
        cv (int, optional): Number of cross-validation folds. Defaults to cv from config file.

    Returns:
        tuple: A tuple containing the trained model, result DataFrame with evaluation metrics,
            cross-validation scores, and DataFrame containing feature importances.
    """
    if UseGridSearch and isinstance(grid_search_dic, dict):
        logger.info(f"Using Grid Search for {model_name}.")
        grid_search = GridSearchCV(model, grid_search_dic[model_name], cv=cv)
        grid_search.fit(X_train, Y_train)
        best_params = grid_search.best_params_
        model.set_params(**best_params)
        if isCat:
            logger.info(f"{model_name} DEFAULT RISK best parameters = {best_params}")
        else:
            logger.info(f"{model_name} LOSS AMOUNT best parameters = {best_params}")
    else:
        logger.info(f"Using Default Parameters for {model_name}.")
        model.set_params(**{"max_depth": max_depth, "random_state": random_state})
    model_ = model.fit(X_train, Y_train)
    Y_Pred_train = model_.predict(X_train)
    Y_Pred_test = model_.predict(X_test)
    if isCat:
        train_accuracy = accuracy_score(Y_train, Y_Pred_train)
        test_accuracy = accuracy_score(Y_test, Y_Pred_test)
        logger.info(f"{model_name} Accuracy Train: {train_accuracy:.4f}")
        logger.info(f"{model_name} Accuracy Test:: {test_accuracy:.4f}")

        def create_results(X, Y, data_type):
            probs = model_.predict_proba(X)
            p = probs[:, 1]
            fpr, tpr, _ = roc_curve(Y, p)
            roc_auc_result = auc(fpr, tpr)
            logger.info(f"{model_name} AUC {data_type}: {roc_auc_result:.3f}")
            dresult = {"fpr": fpr, "tpr": tpr}
            dft = pd.DataFrame(dresult)
            dft["label"] = f"AUC {data_type}: {roc_auc_result:.3f}"
            dft["model_auc"] = f"{model_name} AUC: {roc_auc_result:.3f}"
            return dft

        dftrain = create_results(X_train, Y_train, "Train")
        dftest = create_results(X_test, Y_test, "Test")
        result = pd.concat([dftrain, dftest])
        cv_scores = cross_val_score(model_, X_train, Y_train, scoring="roc_auc", cv=cv)
        logger.info(
            f"{model_name} AUC Train Cross Validation Average: {np.mean(cv_scores):.3f}"
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
            f"{model_name} RMSE Train Cross Validation Average: {np.mean(cv_scores):.3f}"
        )

    importance_df = pd.DataFrame(
        {"Feature": X_train.columns, "Importance": model_.feature_importances_}
    )
    importance_df = importance_df.sort_values(by="Importance", ascending=False)
    importance_df = importance_df[importance_df["Importance"] > 0]
    return model_, result, cv_scores, importance_df


@ignore_warnings(category=ConvergenceWarning)
def process_lr_model(
    model: BaseEstimator,
    model_name: str,
    X_train: pd.DataFrame,
    Y_train: pd.Series,
    X_test: pd.DataFrame,
    Y_test: pd.Series,
    isCat: bool,
    logger: logging.RootLogger,
    cv: int = cv,
) -> Tuple[BaseEstimator, pd.DataFrame, np.ndarray]:
    """
    Process and evaluate a machine learning model.

    Args:
        model (BaseEstimator): The machine learning model to process.
        model_name (str): The name of the model.
        X_train (pd.DataFrame): Training feature data.
        Y_train (pd.Series): Training target data.
        X_test (pd.DataFrame): Testing feature data.
        Y_test (pd.Series): Testing target data.
        isCat (bool): Indicates whether the problem is a classification problem.
        logger (logging.RootLogger): The logger object for logging information.
        cv (int, optional): Number of cross-validation folds. Defaults to cv from config file

    Returns:
        Tuple[BaseEstimator, pd.DataFrame, np.ndarray]:
            - model_ (BaseEstimator): The trained machine learning model.
            - result (pd.DataFrame): Evaluation results.
            - cv_scores (np.ndarray): Cross-validation scores.
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
            logger.info(f"{model_name} variables AUC {data_type}: {roc_auc_result:.3f}")
            dresult = {"fpr": fpr, "tpr": tpr}
            dft = pd.DataFrame(dresult)
            dft["label"] = f"AUC {data_type}: {roc_auc_result:.3f}"
            dft["model_auc"] = f"{model_name} AUC: {roc_auc_result:.3f}"
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


@ignore_warnings(category=ConvergenceWarning)
def process_tf_model(
    model: BaseEstimator,
    model_name: str,
    X_train: pd.DataFrame,
    Y_train: pd.Series,
    X_test: pd.DataFrame,
    Y_test: pd.Series,
    isCat: bool,
    theEpochs: int,
    logger: logging.RootLogger,
    verbose: bool = verbose,
) -> Tuple[object, pd.DataFrame]:
    """ """
    model.fit(X_train, Y_train, epochs=theEpochs, verbose=verbose)
    if isCat:
        Y_Pred_train = np.argmax(model.predict(X_train), axis=1)
        Y_Pred_test = np.argmax(model.predict(X_test), axis=1)
    else:
        Y_Pred_train = model.predict(X_train)
        Y_Pred_test = model.predict(X_test)
    if isCat:
        train_accuracy = accuracy_score(Y_train, Y_Pred_train)
        test_accuracy = accuracy_score(Y_test, Y_Pred_test)
        logger.info(f"{model_name} Accuracy Train: {train_accuracy:.4f}")
        logger.info(f"{model_name} Accuracy Test: {test_accuracy:.4f}")

        def create_results(X, Y, data_type):
            probs = model.predict(X)
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


def getCoef(
    MODEL: BaseEstimator,
    TRAIN_DATA: pd.DataFrame,
    isCat: bool,
    logger: logging.RootLogger,
) -> None:
    """
    Get and log the coefficients of a machine learning model.

    Args:
        MODEL (BaseEstimator): The trained machine learning model.
        TRAIN_DATA (pd.DataFrame): The training data used for the model.
        isCat (bool): Indicates whether the problem is a classification problem.
        logger (logging.RootLogger): The logger object for logging information.

    Returns:
        None
    """
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


def get_important_features(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    Y_train: pd.Series,
    feature_thresh: float = feature_thresh,
    max_depth: int = max_depth,
    random_state: int = random_state,
) -> list:
    """
    Get a list of important features based on a machine learning model's feature importances.

    Args:
        model (BaseEstimator): The machine learning model.
        X_train (pd.DataFrame): Training feature data.
        Y_train (pd.Series): Training target data.
        feature_thresh (float, optional): The feature importance threshold. Defaults to feature_thresh from config file.
        max_depth (int, optional): The maximum depth of the model (if applicable). Defaults to max_depth from config file.
        random_state (int, optional): Random seed for reproducibility. Defaults to random_state from config file.

    Returns:
        list: A list of important feature names.
    """
    model.set_params(**{"max_depth": max_depth, "random_state": random_state})
    model_ = model.fit(X_train, Y_train)
    importance_feat = X_train.columns[model_.feature_importances_ > feature_thresh]
    return importance_feat


@ignore_warnings(category=ConvergenceWarning)
def get_step_wise_features(
    model: BaseEstimator,
    model_type: str,
    logger: logging.RootLogger,
    X_train: pd.DataFrame,
    Y_train: pd.Series,
    img_path: str,
    random_state: int = random_state,
) -> list:
    """
    Perform stepwise feature selection and return the selected features.

    Args:
        model (BaseEstimator): The machine learning model.
        model_type (str): The type or name of the model.
        logger (logging.RootLogger): The logger object for logging information.
        X_train (pd.DataFrame): Training feature data.
        Y_train (pd.Series): Training target data.
        img_path (str): The path for saving a plot (if applicable).
        random_state (int, optional): Random seed for reproducibility. Defaults to random_state from config file.

    Returns:
        list: A list of selected features.
    """
    model.fit(X_train, Y_train)
    fig, ax = plot_sfs(metric_dict=model.get_metric_dict(), kind=None, figsize=figsize)
    ax.set_title(f"{model_type}: Sequential Forward Selection")
    save_img_path = (
        img_path
        + f"{model_type}-Stepwise-rs{random_state}-md{max_depth}-ft{feature_thresh}-mv{max_step_wise_vars}.png"
    )
    fig.savefig(save_img_path, bbox_inches="tight")
    dfm = pd.DataFrame.from_dict(model.get_metric_dict()).T
    dfm = dfm[["feature_names", "avg_score"]]
    dfm.avg_score = dfm.avg_score.astype(float)
    maxIndex = dfm.avg_score.argmax()
    logger.info(f"{model_type} best score: {dfm['avg_score'].iloc[ maxIndex ]} ")
    stepVars = dfm.iloc[maxIndex,]
    stepVars = list(stepVars.feature_names)
    return stepVars


def get_units(size: int) -> int:
    """Creates unit size for neural network.

    Args:
        size (int): Variable size.

    Returns:
        int: Unit size
    """
    return int(2 * size)


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


def plot_bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    title: str,
    save_img_path: str = None,
    figsize: tuple = figsize,
) -> None:
    """Function to save graph

    Args:
        df (pd.DataFrame): Dataframe used for plotting.
        x (str): x variable column for barplot.
        y (str): y variable column for barplot.
        hue (str): hue variable column for barplot.
        title (str): Title for the graph.
        save_img_path (str, optional): File path to save the plot as an image. Defaults to None.
        figsize (tuple, optional): Figure size. Defaults to figsize from config file.
    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=figsize)
    a = sns.barplot(ax=ax, x=x, y=y, hue=hue, data=df)
    a.set(title=title)
    a.grid(axis="y")
    if save_img_path:
        fig.savefig(save_img_path, bbox_inches="tight")


def plot_roc(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    title: str,
    xlabel: str,
    ylabel: str,
    save_img_path: str = None,
    figsize: tuple = figsize,
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
        figsize (tuple, optional): Figure size. Defaults to figsize from config file.
        legend_outside (bool, optional): Flag variable to move legend outside. Defaults to False.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.lineplot(data=data, x=x, y=y, ax=ax, hue=hue)
    sns.lineplot(x=[0, 1], y=[0, 1], ax=ax, linestyle="--", color="gray")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    if legend_outside:
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
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


def plot_decision_tree(
    model: BaseEstimator,
    feature_names: list,
    save_img_path: str = None,
    figsize: tuple = (30, 8),  # (150, 12)
) -> None:
    """
    Plots the decision tree of a scikit-learn model.

    Parameters:
    - model (BaseEstimator): The scikit-learn decision tree model.
    - feature_names (list): A list of feature names.
    - save_img_path (str, optional): The path to save the plotted tree image. If None, the image is not saved.
    - figsize (tuple, optional): A tuple specifying the figure size (width, height) in inches. Default is (30, 8).

    Returns:
    - None
    """
    fig, ax = plt.subplots(figsize=figsize)
    tree.plot_tree(model, feature_names=feature_names, filled=True, ax=ax, rounded=True)
    if save_img_path:
        fig.savefig(save_img_path, bbox_inches="tight")
