import os
import sys
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.functions import save_graph
from src.log_functions import build_logger, get_log_file_name, log_execution_time

current_dir = os.path.dirname(os.path.realpath(__file__))
base_file_name = os.path.splitext(os.path.basename(__file__))[0]
path = os.path.abspath(os.path.join(current_dir, os.pardir))

log_path = path + f"/logs/"
os.makedirs(log_path, exist_ok=True)

# Create logs
log_file_name = get_log_file_name(base_file_name + "-")
log_file_path = log_path + log_file_name
logger = build_logger(log_file_path)

# Read data
data_path = path + "/data/"
df = pd.read_csv(data_path + "HMEQ_LOSS.csv")
TARGET_F = "TARGET_BAD_FLAG"
TARGET_A = "TARGET_LOSS_AMT"
targetcols = [TARGET_F, TARGET_A]
objcols = ["REASON", "JOB"]
mask_numcols = ~df.columns.isin(objcols)
numcols = df.columns[mask_numcols]


# ------- Data Exploration -------
@log_execution_time(logger=logger)
def main():
    logger.info("Info")
    info_output_buffer = StringIO()
    df.info(buf=info_output_buffer)
    info_output_str = info_output_buffer.getvalue()
    logger.info(info_output_str)

    logger.info("Describe")
    desc = df.describe().T.to_string()
    logger.info(f"\n{desc}")

    logger.info("Value Counts")
    for col in objcols:
        vc = df[col].value_counts(dropna=False).to_string()
        logger.info(f"\n{vc}")

    logger.info("Kurtosis")
    ku = df[numcols].kurtosis()
    logger.info(f"\n{ku}")
    logger.info("Skewness")
    sk = df[numcols].skew()
    logger.info(f"\n{sk}")

    logger.info("Describe grouped by each category")
    for col in objcols:
        for numcol in numcols:
            logger.info(f"{col}: {numcol}")
            gdesc = df.groupby(col, dropna=False)[numcol].describe().T.to_string()
            logger.info(f"\n{gdesc}")

    logger.info("Describe grouped by both categories")
    for numcol in numcols:
        logger.info(f"{objcols}: {numcol}")
        gbdesc = df.groupby(objcols, dropna=False)[numcol].describe().T.to_string()
        logger.info(f"\n{gbdesc}")

    logger.info("Save distribution figure of numerical variables.")
    img_path = path + f"/img/{base_file_name}/"
    os.makedirs(img_path, exist_ok=True)
    df.hist(bins=20, figsize=(12, 14))
    fig_save_path = img_path + "distribution_num_variables.png"
    plt.savefig(fig_save_path)
    logger.info(f"Distribution figure save location: {fig_save_path}")

    logger.info(
        "Save barchart of the average default rate and loss amount by job and reason of loan."
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
        logger.info(f"Saved figure to {save_img_path}")
    logger.info("")
    logger.info("Expected value for default considering default rate and loss amount")
    expected_df = (
        df.groupby(objcols)
        .agg(
            default_mean=(TARGET_F, "mean"),
            loss_mean=(TARGET_A, "mean"),
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
    logger.info(f"Saved figure to {save_img_path_exploss}")

    logger.info("Counts of the observation for each job and loan type.")
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
    logger.info(f"Saved figure to {save_img_path_obs}")

    logger.info("Correlation for bad flag.")
    cobf = (
        df[numcols]
        .corr()[TARGET_F]
        .reset_index()
        .sort_values(TARGET_F, ascending=False)
        .query("TARGET_BAD_FLAG < 1 ")
        .to_string()
    )
    logger.info(f"\n{cobf}")

    logger.info("Correlation for bad flag grouped by job and reason for loan.")
    cobfjr = (
        df.groupby(objcols)[numcols]
        .corr()[TARGET_F]
        .reset_index()
        .sort_values(TARGET_F, ascending=False)
        .query("TARGET_BAD_FLAG < 1 ")
        .to_string()
    )
    logger.info(f"\n{cobfjr}")
    logger.info("Completed")

    print("Completed!")
    print(f"Saved graph images to {img_path}")
    print(f"Saved logs for analysis to {log_file_path}")


if __name__ == "__main__":
    main()
