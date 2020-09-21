# -*- coding: utf-8 -*-
#!/user/bin/env python3
"""
@author:U Plan Team 2
@description: U Plan final project Main Function
"""

import pandas as pd
import data_preprocess
import xgboost_churn
import lightGBM_churn
import randomForest_churn
import dnn_churn
import drawRoc


def main(path):
    # read raw data
    churn_raw_data = pd.read_csv(path)

    # preprocessing data
    X_train, X_test, y_train, y_test = data_preprocess.preprocess(churn_raw_data)

    # xgboost model training
    xgb_fpr, xgb_tpr, xgb_roc_auc = xgboost_churn.xgboost_churn(X_train, y_train, X_test, y_test)

    # lightGBM model training
    gbm_fpr, gbm_tpr, gbm_roc_auc = lightGBM_churn.lightGBM_churn(X_train, y_train, X_test, y_test)

    # randomForest model training
    rf_fpr, rf_tpr, rf_roc_auc = randomForest_churn.random_forest_churn(X_train, y_train, X_test, y_test)

    # DNN model training
    dnn_fpr, dnn_tpr, dnn_roc_auc = dnn_churn.dnn_churn(X_train, y_train, X_test, y_test)

    # plot ROC
    drawRoc.drawRoc([xgb_fpr, xgb_tpr, xgb_roc_auc], [gbm_fpr, gbm_tpr, gbm_roc_auc],
                    [rf_fpr, rf_tpr, rf_roc_auc], [dnn_fpr, dnn_tpr, dnn_roc_auc])


if __name__ == '__main__':
    data_path = "./data/cell2celltrain.csv"
    main(data_path)
