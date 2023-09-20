import os
import sys
import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn import tree


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.functions import (
    preprocess_data_with_log,
    split_data,
    process_lr_model,
    getCoef,
    get_important_features,
    get_step_wise_features,
    plot_roc,
)
from src.config import (
    TARGET_F,
    TARGET_A,
    targetcols,
    objcols,
    numcols,
    random_state,
    cv,
    max_depth,
    feature_thresh,
    max_step_wise_vars,
    figsize,
)
from src.log_functions import build_logger, get_log_file_name, log_execution_time

# -------- Configure output path and logging --------
current_dir = os.path.dirname(os.path.realpath(__file__))
base_file_name = os.path.splitext(os.path.basename(__file__))[0]
path = os.path.abspath(os.path.join(current_dir, os.pardir))
log_path = path + f"/logs/"
img_path = path + f"/img/{base_file_name}/"
os.makedirs(log_path, exist_ok=True)
os.makedirs(img_path, exist_ok=True)

# Create logs
log_file_name = get_log_file_name(base_file_name + "-")
log_file_path = log_path + log_file_name
logger = build_logger(log_file_path)


@log_execution_time(logger=logger)
def main():
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
            name_str = name.replace(" ", "-")
            if model_type == "Categorical":
                isCat = True
                if name in ["Decision Tree", "Random Forest", "Gradient Boosting"]:
                    importance_feat = get_important_features(
                        model=model,
                        X_train=X_train,
                        Y_train=Y_train,
                        feature_thresh=feature_thresh,
                        max_depth=max_depth,
                        random_state=random_state,
                    )
                    logger.info(f"Important Variables from {name}: {importance_feat}")
                    X_train_ = X_train.loc[:, importance_feat].copy()
                    X_test_ = X_test.loc[:, importance_feat].copy()
                elif name == "Step Wise":
                    importance_feat = get_step_wise_features(
                        model=model,
                        model_type=model_type,
                        logger=logger,
                        X_train=X_train,
                        Y_train=Y_train,
                        img_path=img_path,
                        random_state=random_state,
                    )
                    logger.info(f"Important Variables from {name}: {importance_feat}")
                    X_train_ = X_train.loc[:, importance_feat].copy()
                    X_test_ = X_test.loc[:, importance_feat].copy()
                else:
                    X_train_ = X_train.copy()
                    X_test_ = X_test.copy()

                model_new, result, cv_scores = process_lr_model(
                    model=LogisticRegression(),
                    model_name=name,
                    X_train=X_train_,
                    Y_train=Y_train,
                    X_test=X_test_,
                    Y_test=Y_test,
                    isCat=isCat,
                    logger=logger,
                    cv=cv,
                )
                getCoef(model_new, X_train_, isCat, logger=logger)
                titlec = f"Logistic Regression ROC Curve ({name} variables)"
                save_img_path = (
                    img_path
                    + f"ROC-{name_str}-features--rs{random_state}-cv{cv}-md{max_depth}-ft{feature_thresh}-mv{max_step_wise_vars}.png"
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
                        feature_thresh=feature_thresh,
                        max_depth=max_depth,
                        random_state=random_state,
                    )
                    logger.info(f"Important Variables from {name}: {importance_feat}")
                    XA_train_ = XA_train.loc[:, importance_feat].copy()
                    XA_test_ = XA_test.loc[:, importance_feat].copy()
                elif name == "Step Wise":
                    importance_feat = get_step_wise_features(
                        model=model,
                        model_type=model_type,
                        logger=logger,
                        X_train=XA_train,
                        Y_train=YA_train,
                        img_path=img_path,
                        random_state=random_state,
                    )
                    logger.info(f"Important Variables from {name}: {importance_feat}")
                    XA_train_ = XA_train.loc[:, importance_feat].copy()
                    XA_test_ = XA_test.loc[:, importance_feat].copy()
                else:
                    XA_train_ = XA_train.copy()
                    XA_test_ = XA_test.copy()

                model_new, result, cv_scores = process_lr_model(
                    model=LinearRegression(),
                    model_name=name,
                    X_train=XA_train_,
                    Y_train=YA_train,
                    X_test=XA_test_,
                    Y_test=YA_test,
                    isCat=isCat,
                    logger=logger,
                    cv=cv,
                )
                getCoef(model_new, XA_train_, isCat, logger=logger)
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


if __name__ == "__main__":
    main()
