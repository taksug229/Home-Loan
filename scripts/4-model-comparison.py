import os
import sys
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.functions import (
    preprocess_data_with_log,
    split_data,
    process_model,
    process_lr_model,
    process_tf_model,
    get_important_features,
    get_units,
    scale_data,
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
    feature_thresh_dl,
    variable_selection_model,
    activations_names,
    verbose,
    epochs_categorical,
    epochs_regression,
    drop_out_ratio,
    figsize,
)
from src.log_functions import build_logger, get_log_file_name, log_execution_time

# -------- Configure output path and logging --------
current_dir = os.path.dirname(os.path.realpath(__file__))
base_file_name = os.path.splitext(os.path.basename(__file__))[0]
path = os.path.abspath(os.path.join(current_dir, os.pardir))
if not variable_selection_model:
    variable_selection_model = "All"

img_path = path + f"/img/{base_file_name}/{variable_selection_model}_vars/"
log_path = path + f"/logs/{base_file_name}/{variable_selection_model}_vars/"
os.makedirs(img_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)

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

    # -------- Scale Data for TensorFlow --------
    theScaler = MinMaxScaler()
    theScaler.fit(X_train)
    X_train_tf = scale_data(X=X_train, theScaler=theScaler)
    X_test_tf = scale_data(X=X_test, theScaler=theScaler)
    XA_train_tf = scale_data(X=XA_train, theScaler=theScaler)
    XA_test_tf = scale_data(X=XA_test, theScaler=theScaler)

    # -------- Model Settings --------
    model_dic = {
        "LR": {
            "Categorical": LogisticRegression(),
            "Regression": LinearRegression(),
        },
        "DT": {
            "Categorical": tree.DecisionTreeClassifier(),
            "Regression": tree.DecisionTreeRegressor(),
        },
        "RF": {
            "Categorical": RandomForestClassifier(),
            "Regression": RandomForestRegressor(),
        },
        "GB": {
            "Categorical": GradientBoostingClassifier(),
            "Regression": GradientBoostingRegressor(),
        },
    }

    # -------- Variable selection --------
    if variable_selection_model == "All":
        logger.info("Using all variables")
        logger.info(f"All Variables Count: {X_train.shape[1]}")
        logger.info(f"All Variables: {list(X_train.columns.values)}")
    else:
        logger.info(f"Variables Selection enabled for {variable_selection_model}")
        importance_feat_class = get_important_features(
            model=model_dic[variable_selection_model]["Categorical"],
            X_train=X_train,
            Y_train=Y_train,
            feature_thresh=feature_thresh_dl,
            max_depth=max_depth,
            random_state=random_state,
        )
        logger.info(
            f"Important Variables Count from {variable_selection_model} Categorical: {len(importance_feat_class)}"
        )
        logger.info(
            f"Important Variables from {variable_selection_model} Categorical: {importance_feat_class}"
        )
        importance_feat_regress = get_important_features(
            model=model_dic[variable_selection_model]["Regression"],
            X_train=XA_train,
            Y_train=YA_train,
            feature_thresh=feature_thresh_dl,
            max_depth=max_depth,
            random_state=random_state,
        )
        logger.info(
            f"Important Variables from {variable_selection_model} Regression: {len(importance_feat_regress)}"
        )
        logger.info(
            f"Important Variables from {variable_selection_model} Regression: {importance_feat_regress}"
        )
        X_train = X_train.loc[:, importance_feat_class].copy()
        X_test = X_test.loc[:, importance_feat_class].copy()
        XA_train = XA_train.loc[:, importance_feat_regress].copy()
        XA_test = XA_test.loc[:, importance_feat_regress].copy()

    # -------- Plotting Parameters and Labels for ROCAUC --------
    xc = "fpr"
    yc = "tpr"
    huec = "label"
    xlabelc = "False Positive Rate"
    ylabelc = "True Positive Rate"

    # -------- Model Iteration --------
    roc_results_tree = {}
    rmse_results_tree = {}
    for idx, (name, model_types) in enumerate(model_dic.items()):
        for model_type, model in model_types.items():
            logger.info(f"{name} {model_type}")
            if model_type == "Categorical":
                isCat = True
                if name == "LR":
                    _, result, _ = process_lr_model(
                        model=model,
                        model_name=name,
                        X_train=X_train,
                        Y_train=Y_train,
                        X_test=X_test,
                        Y_test=Y_test,
                        isCat=isCat,
                        logger=logger,
                        cv=cv,
                    )
                else:
                    _, result, _, _ = process_model(
                        model=model,
                        model_name=name,
                        X_train=X_train,
                        Y_train=Y_train,
                        X_test=X_test,
                        Y_test=Y_test,
                        isCat=isCat,
                        logger=logger,
                    )
                titlec = f"{name} ROC Curve ({variable_selection_model} Parameters)"
                save_img_path = (
                    img_path
                    + f"ROC-{name}-var-{variable_selection_model}-rs{random_state}-md{max_depth}-ft{feature_thresh_dl}.png"
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
                roc_results_tree[name] = result
            else:
                isCat = False
                if name == "LR":
                    _, result, _ = process_lr_model(
                        model=model,
                        model_name=name,
                        X_train=XA_train,
                        Y_train=YA_train,
                        X_test=XA_test,
                        Y_test=YA_test,
                        isCat=isCat,
                        logger=logger,
                        cv=cv,
                    )
                else:
                    _, result, _, _ = process_model(
                        model=model,
                        model_name=name,
                        X_train=XA_train,
                        Y_train=YA_train,
                        X_test=XA_test,
                        Y_test=YA_test,
                        isCat=isCat,
                        logger=logger,
                    )
                rmse_results_tree[name] = result
            logger.info("------------------------------------")

    # -------- TensorFlow Iteration --------
    tf_model_types = ["Categorical", "Regression"]
    hidden_layers = [1, 2]
    dropout_layers = [True, False]
    roc_results_tf = {}
    rmse_results_tf = {}
    for act_name, activation in activations_names.items():
        for hlayers in hidden_layers:
            for dropout in dropout_layers:
                for model_type in tf_model_types:
                    if model_type == "Categorical":
                        isCat = True
                        theShapeSize = X_train_tf.shape[1]
                        theLossMetric = tf.keras.losses.SparseCategoricalCrossentropy()
                        LAYER_OUTPUT = tf.keras.layers.Dense(
                            units=2, activation=tf.keras.activations.softmax
                        )
                        theEpochs = epochs_categorical
                    else:
                        isCat = False
                        theShapeSize = XA_train_tf.shape[1]
                        theLossMetric = tf.keras.losses.MeanSquaredError()
                        LAYER_OUTPUT = tf.keras.layers.Dense(
                            units=1, activation=tf.keras.activations.linear
                        )
                        theEpochs = epochs_regression
                    name = f"TF-{act_name}-hl-{hlayers}-do-{dropout}-ep-{theEpochs}"
                    theOptimizer = tf.keras.optimizers.Adam()
                    theUnits = get_units(theShapeSize)
                    LAYER_01 = tf.keras.layers.Dense(
                        units=theUnits, activation=activation, input_dim=theShapeSize
                    )
                    LAYER_DROP = tf.keras.layers.Dropout(drop_out_ratio)
                    LAYER_02 = tf.keras.layers.Dense(
                        units=theUnits, activation=activation
                    )

                    TFM = tf.keras.Sequential()
                    TFM.add(LAYER_01)
                    if dropout:
                        TFM.add(LAYER_DROP)
                    if hlayers == 2:
                        TFM.add(LAYER_02)
                        if dropout:
                            TFM.add(LAYER_DROP)
                    TFM.add(LAYER_OUTPUT)
                    TFM.compile(loss=theLossMetric, optimizer=theOptimizer)
                    if model_type == "Categorical":
                        model_new, result = process_tf_model(
                            model=TFM,
                            model_name=name,
                            X_train=X_train_tf,
                            Y_train=Y_train,
                            X_test=X_test_tf,
                            Y_test=Y_test,
                            isCat=isCat,
                            theEpochs=theEpochs,
                            logger=logger,
                            verbose=verbose,
                        )
                        titlec = (
                            f"{name}: ROC Curve ({variable_selection_model} Parameters)"
                        )
                        save_img_path = (
                            img_path
                            + f"ROC-{name}-var-{variable_selection_model}-rs{random_state}-md{max_depth}-ft{feature_thresh_dl}.png"
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
                        roc_results_tf[f"{titlec}"] = result
                    else:
                        model_new, result = process_tf_model(
                            model=TFM,
                            model_name=name,
                            X_train=XA_train_tf,
                            Y_train=YA_train,
                            X_test=XA_test_tf,
                            Y_test=YA_test,
                            isCat=isCat,
                            theEpochs=theEpochs,
                            logger=logger,
                            verbose=verbose,
                        )
                        rmse_results_tf[name] = result
                    logger.info("------------------------------------")

    # -------- Comparing all results and saving --------
    roc_df_tf = pd.concat([d for d in roc_results_tf.values()])
    roc_df_tf = roc_df_tf[roc_df_tf["label"].str.contains("Test")].reset_index(
        drop=True
    )
    rmse_df_tf = pd.concat([d for d in rmse_results_tf.values()])
    rmse_df_tf = rmse_df_tf[rmse_df_tf["label"].str.contains("Test")].reset_index(
        drop=True
    )
    save_img_path_tf = (
        img_path
        + f"ROC-TF-Comparison--ep{theEpochs}-var-{variable_selection_model}-rs{random_state}-md{max_depth}-ft{feature_thresh_dl}.png"
    )
    title_comapre_tf = (
        f"ROC Curve Tensor Flow Comparison ({variable_selection_model} Parameters)"
    )
    plot_roc(
        data=roc_df_tf,
        x=xc,
        y=yc,
        hue="model_auc",
        title=title_comapre_tf,
        xlabel=xlabelc,
        ylabel=ylabelc,
        save_img_path=save_img_path_tf,
        figsize=figsize,
        legend_outside=True,
    )

    roc_df_tree = pd.concat([d for d in roc_results_tree.values()])
    roc_df_tree = roc_df_tree[roc_df_tree["label"].str.contains("Test")].reset_index(
        drop=True
    )
    rmse_df_tree = pd.concat([d for d in rmse_results_tree.values()])
    rmse_df_tree = rmse_df_tree[rmse_df_tree["label"].str.contains("Test")].reset_index(
        drop=True
    )
    roc_df_tf_best = roc_df_tf[
        (roc_df_tf["label"].str.contains("Test"))
        & (roc_df_tf["auc"] == roc_df_tf["auc"].max())
    ].reset_index(drop=True)

    roc_df_comparison = pd.concat([roc_df_tree, roc_df_tf_best]).reset_index(drop=True)
    rmse_df_comparison = pd.concat([rmse_df_tree, rmse_df_tf]).reset_index(drop=True)

    save_img_path_comparison = (
        img_path
        + f"ROC-Comparison--ep{theEpochs}-var-{variable_selection_model}-rs{random_state}-md{max_depth}-ft{feature_thresh_dl}.png"
    )
    title_comapre_all = f"ROC Curve Comparison ({variable_selection_model} Parameters)"
    plot_roc(
        data=roc_df_comparison,
        x=xc,
        y=yc,
        hue="model_auc",
        title=title_comapre_all,
        xlabel=xlabelc,
        ylabel=ylabelc,
        save_img_path=save_img_path_comparison,
        figsize=figsize,
    )
    rmse_df_comparison = rmse_df_comparison.sort_values("RMSE").reset_index(drop=True)
    save_df_path = (
        log_path
        + f"RMSE-Comparison--ep{theEpochs}-var-{variable_selection_model}-rs{random_state}-md{max_depth}-ft{feature_thresh_dl}.csv"
    )
    rmse_df_comparison.to_csv(save_df_path, index=False)
    print("\nCompleted!")
    print(f"Saved images to {img_path}")
    print(f"Saved logs for analysis to {log_file_path}")


if __name__ == "__main__":
    main()
