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


def main(path):
    # read raw data
    churn_raw_data = pd.read_csv(path)

    # preprocessing data
    X_train, y_train, X_test, y_test = data_preprocess.preprocess(churn_raw_data)

    # xgboost model training
    xgboost_churn.xgboost_churn(X_train, y_train, X_test, y_test)

    # lightGBM model training
    lightGBM_churn.lightGBM_churn(X_train, y_train, X_test, y_test)

    # randomForest model training
    randomForest_churn.random_forest_churn(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    data_path = "data/cell2celltrain.csv"
    main(data_path)
