import pandas as pd
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns


def data_imputation(
    df: pd.DataFrame,
    TARGET_F: str = "TARGET_BAD_FLAG",
    TARGET_A: str = "TARGET_LOSS_AMT",
    objcols: List[str] = ["REASON", "JOB"],
) -> pd.DataFrame:
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
    data = df.copy()
    for col in objcols:
        thePrefix = "z_" + col
        y = pd.get_dummies(data[col], prefix=thePrefix, drop_first=False)
        y = y.drop(y.columns[-1], axis=1).astype(int)
        data = pd.concat([data, y], axis=1)
        data = data.drop(col, axis=1)
    return data


def data_cap(df: pd.DataFrame, TARGET_A: str = "TARGET_LOSS_AMT", cap: int = 25_000):
    data = df.copy()
    cap_limit = data[TARGET_A] > cap
    data.loc[cap_limit, TARGET_A] = cap
    return


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
