# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:57:00 2019
Instead of Cross Validation, this Tunning function use only one out-of-sample validation.
This should reduce the tunning time, but still have some benefits of the validation.
"""
import numpy as np
import lightgbm as lgb
from sklearn import metrics, datasets
from sklearn.model_selection import train_test_split
import itertools as it

def metric_function(labels, pred, metric = 'auc'):
    if metric == "auc":
        fpr, tpr, thresholds = metrics.roc_curve(labels, pred)
        metric_value = metrics.auc(fpr, tpr)
    return metric_value

def LGBMTuning(
    params,
    X_train,
    y_train,
    X_dev=None,
    y_dev=None,
    objective="binary",
    metric="auc",
    num_iterations=[100],
    early_stopping_rounds=None,
    verbose_eval=False,
):
    # initilize the lightgbm parameter
    lgb_params = {
        "objective": [objective],
        "metric": [metric],
        "num_leaves": [32],
        "max_depth": [1],
        "min_data_in_leaf": [20],
        "learning_rate": [0.1],
        "bagging_fraction": [0.5],
        "top_rate": [0.2],
        "lambda_l1": [0],
        "lambda_l2": [0],
        "min_gain_to_split": [0],
        "max_bin": [255],
    }
    lgb_params.update(params)

    # convert dataset to lightgbm dataset
    lgb_train = lgb.Dataset(X_train, label=y_train)
    if (X_dev is not None) & (y_dev is not None):
        lgb_dev = lgb.Dataset(X_dev, label=y_dev)
        use_dev = True
    else:
        lgb_dev = None
        use_dev = False

    # Choose to Maximize or Minimize Metric Function Value
    if metric in ["auc"]:
        metric_eval = "Maximize"
        best_value = -np.exp(20)
    elif metric in ["binary_logloss", "binary"]:
        metric_eval = "Minimize"
        best_value = np.exp(20)

    # Get All Grid Combinations
    params_names = list(lgb_params.keys())
    params_combinations = it.product(*(lgb_params[name] for name in params_names))
    params_combinations = list(params_combinations)

    for n in num_iterations:
        for params_iter in params_combinations:
            model_param = dict(zip(params_names, params_iter))
            lgb_model = lgb.train(
                model_param,
                train_set=lgb_train,
                num_boost_round=n,
                valid_sets=lgb_dev,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=verbose_eval,
            )
            # Use dev if dev exists, else use train in the validation metric
            if use_dev:
                pred = lgb_model.predict(
                    data=X_dev, num_iteration=lgb_model.best_iteration
                )
                labels = y_dev
            else:
                pred = lgb_model.predict(
                    data=X_train, num_iteration=lgb_model.best_iteration
                )
                labels = y_train

            metric_value = metric_function(labels, pred, metric = metric)

            if metric_eval == "Maximize":
                if metric_value > best_value:
                    best_param, best_n, best_value, best_model = model_param, n, metric_value, lgb_model
                    print(
                        "Best Parameters Update: {} Num of Iteration: {} Best Value: {} \n".format(
                            best_param, best_n, best_value
                        )
                    )

            if metric_eval == "Minimize":
                if metric_value < best_value:
                    best_param, best_n, best_value, best_model = model_param, n, metric_value, lgb_model
                    print(
                        "Best Parameters Update: {} Num of Iteration: {} Best Value: {} \n".format(
                            best_param, best_n, best_value
                        )
                    )
    print(
        "\n Best Parameters Final: {} Num of Iteration: {} Best Value: {} \n".format(
            best_param, best_n, best_value, best_model
        )
    )
    return best_param, best_n, best_value, lgb_model


# Example
if __name__ == "__main__":
    cancer = datasets.load_breast_cancer()
    
    X_train = cancer.data
    y_train = cancer.target
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_train, y_train, test_size=0.2, random_state=3154
    )
    
    params = {
        "num_leaves": [16, 32, 64],
        "max_depth": [1, 3, 5],
        "min_data_in_leaf": [10, 20, 50],
        "learning_rate": [0.01, 0.05, 0.1],
        "bagging_fraction": [1.0],
        "top_rate": [0.2],
    }
    num_iterations = [100, 500]
    
    best_param, best_n, best_value, best_model = LGBMTuning(
        params,
        X_train,
        y_train,
        X_dev=X_dev,
        y_dev=y_dev,
        objective="binary",
        metric="auc",
        num_iterations=num_iterations,
        early_stopping_rounds=50,
        verbose_eval=False,
    )
