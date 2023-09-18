import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn import tree
import graphviz

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.functions import (
    preprocess_data_with_log,
    split_data,
    process_model,
    plot_roc,
    plot_importance,
)
from src.config import (
    TARGET_F,
    TARGET_A,
    targetcols,
    objcols,
    numcols,
    random_state,
    cv,
    figsize,
    UseGridSearch,
    title_params,
)
from src.log_functions import build_logger, get_log_file_name, log_execution_time

# -------- Configure output path and logging --------
current_dir = os.path.dirname(os.path.realpath(__file__))
base_file_name = os.path.splitext(os.path.basename(__file__))[0]
path = os.path.abspath(os.path.join(current_dir, os.pardir))
log_path = path + f"/logs/"
img_path = path + f"/img/{base_file_name}/"
viz_path = img_path + f"/viz/"
os.makedirs(log_path, exist_ok=True)
os.makedirs(img_path, exist_ok=True)
os.makedirs(viz_path, exist_ok=True)

# Create logs
log_file_name = get_log_file_name(base_file_name + "-")
log_file_path = log_path + log_file_name
logger = build_logger(log_file_path)

# -------- Reading data, prerpocessing, splitting data. ----------

data_path = path + "/data/"
df = pd.read_csv(data_path + "HMEQ_LOSS.csv")
df = preprocess_data_with_log(
    df=df,
    logger=logger,
    objcols=objcols,
    numcols=numcols,
    targetcols=targetcols,
    TARGET_A=TARGET_A,
)


@log_execution_time(logger=logger)
def main():
    logger.info("LOSS AMOUNT statistics after preprocessing")
    logger.info(f"\n{df[TARGET_A].describe().to_string()}")

    # ----------- Split Data ----------
    (
        X_train,
        X_test,
        Y_train,
        Y_test,
        XA_train,
        XA_test,
        YA_train,
        YA_test,
    ) = split_data(
        df=df,
        logger=logger,
        TARGET_F=TARGET_F,
        TARGET_A=TARGET_A,
        random_state=random_state,
    )

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
                    logger=logger,
                    UseGridSearch=UseGridSearch,
                    cv=cv,
                )
                titlec = f"{name} ROC Curve ({title_params})"
                save_img_path = (
                    img_path + f"ROC-{name}-{title_params}-{random_state}.png"
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
                model_new, result, cv_scores, important_vars = process_model(
                    model=model,
                    model_name=name,
                    X_train=XA_train,
                    Y_train=YA_train,
                    X_test=XA_test,
                    Y_test=YA_test,
                    TARGET=TARGET_A,
                    isCat=isCat,
                    logger=logger,
                    UseGridSearch=UseGridSearch,
                    cv=cv,
                )
            logger.info(
                f'Important Variables\n{important_vars.query("Importance > 0")}'
            )

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
                feature_cols = list(X_train.columns.values)
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


if __name__ == "__main__":
    main()
