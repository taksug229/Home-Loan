import pandas as pd
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import logging


def data_imputation(
    df: pd.DataFrame,
    TARGET_F: str = "TARGET_BAD_FLAG",
    TARGET_A: str = "TARGET_LOSS_AMT",
    objcols: List[str] = ["REASON", "JOB"],
) -> pd.DataFrame:
    """
    Impute missing values in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        TARGET_F (str, optional): The name of the binary target column. Defaults to "TARGET_BAD_FLAG".
        TARGET_A (str, optional): The name of the numeric target column. Defaults to "TARGET_LOSS_AMT".
        objcols (List[str], optional): A list of categorical column names. Defaults to ["REASON", "JOB"].

    Returns:
        pd.DataFrame: The DataFrame with missing values imputed.
    """
    data = df.copy()
    targetcols = [TARGET_F, TARGET_A]
    mask_numcols = ~data.columns.isin(objcols)
    numcols = data.columns[mask_numcols]
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
    objcols: List[str] = ["REASON", "JOB"],
) -> pd.DataFrame:
    """
    Create dummy variables for categorical columns in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        objcols (List[str], optional): A list of categorical column names. Defaults to ["REASON", "JOB"].

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
    df: pd.DataFrame, TARGET_A: str = "TARGET_LOSS_AMT", cap: int = 25_000
) -> pd.DataFrame:
    """
    Cap values in a specific column of a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        TARGET_A (str, optional): The name of the target column to be capped. Defaults to "TARGET_LOSS_AMT".
        cap (int, optional): The value to which the target column values will be capped. Defaults to 25,000.

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
    objcols: List[str],
    numcols: List[str],
    TARGET_F: str = "TARGET_BAD_FLAG",
    TARGET_A: str = "TARGET_LOSS_AMT",
) -> pd.DataFrame:
    """
    Preprocess data with logging.

    Args:
        df (pd.DataFrame): The input DataFrame.
        logger (logging.RootLogger): The logger object for logging.
        objcols (List[str]): A list of categorical column names.
        numcols (List[str]): A list of numerical column names.
        TARGET_F (str, optional): The name of the binary target column. Defaults to "TARGET_BAD_FLAG".
        TARGET_A (str, optional): The name of the numeric target column. Defaults to "TARGET_LOSS_AMT".

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
        df=data, TARGET_F=TARGET_F, TARGET_A=TARGET_A, objcols=objcols
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
    """
    fig, ax = plt.subplots()
    a = sns.barplot(ax=ax, x=x, y=y, hue=hue, data=df)
    a.set(title=title)
    a.grid(axis="y")
    fig.savefig(save_img_path)
