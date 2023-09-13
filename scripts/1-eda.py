import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Full path of the directory of this Python file
path = os.path.dirname(os.path.realpath(__file__))
log_file_path = path + "/" + "analysis_log.txt"
df = pd.read_csv(path + "/" + "HMEQ_LOSS.csv")
targetcols = ["TARGET_BAD_FLAG", "TARGET_LOSS_AMT"]
objcols = ["REASON", "JOB"]
mask_numcols = ~df.columns.isin(objcols)
numcols = df.columns[mask_numcols]


def dfmethod_to_text_save(dfmethod, f):
    """Save Pandas dataframe methods to log file.

    Args:
        dfmethod (None): Pandas dataframe methods.
        f (TextIOWrapper): File to save log file.
    """
    txt = dfmethod.to_string()
    f.write(txt)
    f.write("\n")
    f.write("\n")


def save_graph(df, x, y, hue, title, save_img_path):
    """_summary_

    Args:
        df (pd.DataFrame): Dataframe used for plotting.
        x (str): x variable column for barplot.
        y (str): y variable column for barplot.
        hue (str): hue variable column for barplot.
        save_img_path (str): image save path.
    """
    fig, ax = plt.subplots()
    a = sns.barplot(ax=ax, x=x, y=y, hue=hue, data=df)
    a.set(title=title)
    a.grid(axis="y")
    fig.savefig(save_img_path)


# ------- Data Exploration -------
with open(log_file_path, "w") as f:
    f.write("Info\n")
    df.info(buf=f)
    f.write("\n")

    f.write("Describe\n")
    dfmethod_to_text_save(df.describe().T, f)

    f.write("Value Counts\n")
    for col in objcols:
        dfmethod_to_text_save(df[col].value_counts(dropna=False), f)

    f.write("Kurtosis\n")
    dfmethod_to_text_save(df[numcols].kurtosis(), f)
    f.write("Skewness\n")
    dfmethod_to_text_save(df[numcols].skew(), f)

    f.write("Describe grouped by each category\n")
    for col in objcols:
        for numcol in numcols:
            f.write(f"{col}: {numcol}\n")
            dfmethod_to_text_save(df.groupby(col, dropna=False)[numcol].describe().T, f)
    f.write("\n")

    f.write("Describe grouped by both categories\n")
    for numcol in numcols:
        f.write(f"{objcols}: {numcol}\n")
        dfmethod_to_text_save(df.groupby(objcols, dropna=False)[numcol].describe().T, f)
    f.write("\n")

    f.write("Save distribution figure of numerical variables.\n")
    img_path = path + "/" + "img/"
    os.makedirs(img_path, exist_ok=True)
    df.hist(bins=20, figsize=(12, 14))
    fig_save_path = img_path + "distribution_num_variables.png"
    plt.savefig(fig_save_path)
    f.write(f"Distribution figure save location: {fig_save_path}\n")

    f.write(
        "Save barchart of the average default rate and loss amount by job and reason of loan.\n"
    )
    for tcol in targetcols:
        temp_df = df.groupby(objcols).agg(mean=(tcol, "mean")).reset_index()
        save_img_path = img_path + f"barchart_{tcol}.png"
        save_graph(
            df=temp_df,
            x="JOB",
            y="mean",
            hue="REASON",
            title=tcol,
            save_img_path=save_img_path,
        )
        f.write(f"Saved figure to {save_img_path}\n")
    f.write("\n")
    f.write("Expected value for default considering default rate and loss amount\n")
    expected_df = (
        df.groupby(objcols)
        .agg(
            default_mean=("TARGET_BAD_FLAG", "mean"),
            loss_mean=("TARGET_LOSS_AMT", "mean"),
        )
        .reset_index()
    )
    expected_df["expected_loss_amount"] = (
        expected_df["default_mean"] * expected_df["loss_mean"]
    )
    yloss = "expected_loss_amount"
    save_img_path_exploss = img_path + f"barchart_{yloss}.png"
    save_graph(
        df=expected_df,
        x="JOB",
        y=yloss,
        hue="REASON",
        title=yloss,
        save_img_path=save_img_path_exploss,
    )
    f.write(f"Saved figure to {save_img_path_exploss}\n")
    f.write("\n")

    f.write("Counts of the observation for each job and loan type.\n")
    count_df = df.groupby(objcols).size().reset_index()
    count_df = count_df.rename(columns={0: "Observations"})
    yobs = "Observations"
    save_img_path_obs = img_path + f"barchart_{yobs}.png"
    save_graph(
        df=count_df,
        x="JOB",
        y=yobs,
        hue="REASON",
        title=yobs,
        save_img_path=save_img_path_obs,
    )
    f.write(f"Saved figure to {save_img_path_obs}\n")
    f.write("\n")

    f.write("Correlation for bad flag.\n")
    dfmethod_to_text_save(
        df[numcols]
        .corr()["TARGET_BAD_FLAG"]
        .reset_index()
        .sort_values("TARGET_BAD_FLAG", ascending=False)
        .query("TARGET_BAD_FLAG < 1 "),
        f,
    )

    f.write("Correlation for bad flag grouped by job and reason for loan.\n")
    dfmethod_to_text_save(
        df.groupby(objcols)[numcols]
        .corr()["TARGET_BAD_FLAG"]
        .reset_index()
        .sort_values("TARGET_BAD_FLAG", ascending=False)
        .query("TARGET_BAD_FLAG < 1 "),
        f,
    )

    # ------- Impute Missing Values -------

    f.write("Total NaN values in each row.\n")
    f.write(
        "NOTE: I am not imputing TARGET_LOSS_AMT because this is only NaN when the person did not default.\n"
    )
    dfmethod_to_text_save(df.isna().sum(), f)

    f.write(
        "Median of each numerical column grouped by job and reason of loan. I will impute NaNs with these values.\n"
    )
    dfmethod_to_text_save(df.groupby(objcols, dropna=False)[numcols].median(), f)

    f.write("Impute missing values START.\n")
    f.write("Impute missing categorical values to 'Unknwon'.\n")
    for col in objcols:
        df.loc[df[col].isna(), col] = "Unknown"

    f.write("Add new column for flag and imputed numbers.\n")
    f.write("The NaNs will be imputed by the median of per Job and Reason.\n")
    f.write("The original imputed columns will be deleted.\n")
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
    f.write("Impute missing values END.\n")
    f.write("\n")
    f.write("First 5 observations of the imputed data\n")
    dfmethod_to_text_save(df.head().T, f)
    f.write(
        "Confirming there are no NaNs (except the intentionally left TARGET_LOSS_AMT.)\n"
    )
    dfmethod_to_text_save(df.isna().sum(), f)

    # ------- Create dummy variables -------
    f.write("Creating dummy variables for categorical columns\n")
    for col in objcols:
        thePrefix = "z_" + col
        y = pd.get_dummies(df[col], prefix=thePrefix, drop_first=False)
        y = y.drop(y.columns[-1], axis=1).astype(int)
        df = pd.concat([df, y], axis=1)
        df = df.drop(col, axis=1)
    f.write("Completed creating dummy variables.\n")
    f.write("First 5 observations of the data with dummy variables\n")
    dfmethod_to_text_save(df.head(), f)
    f.write("Saving data\n")
    save_csv_path = path + "/" + "NEW_HMEQ_LOSS.csv"
    df.to_csv(save_csv_path, index=False)
    f.write("Completed\n")
    f.close()
    print("Completed!\n")
    print(f"Saved graph images to {img_path}")
    print(f"Saved processed csv to {save_csv_path}")
    print(f"Saved logs for analysis to {log_file_path}")
