# -*- coding: utf-8 -*-
#!/user/bin/env python3
"""
@author:U Plan Team 2
@description: U Plan final project Main Function
"""

import sys
import data_preprocess
import numpy as np
import matplotlib as plt
import pandas as pd


def main(path):
    # read raw data
    churn_raw_data = pd.read_csv(path)

    # preprocessing data
    X_train, y_train, X_test, y_test = data_preprocess.preprocess(churn_raw_data)

    # # model training
    # xgb_pred_label =


if __name__ == '__main__':
    data_path = "data/cell2celltrain.csv"
    main(data_path)
