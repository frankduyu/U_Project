"""
@author:U Plan Team 2
@description: U Plan final project Main Function
"""

import sys
import data_preprocess
import train_test_split
import numpy as np
import matplotlib as plt
import pandas as pd


def main(path):
    # read raw data
    churn_data = pd.read_csv(path)

    # preprocessing data
    churn_preproed_data = data_preprocess.preprocess(churn_data)

    # split train & test data
    # train_data, test_data = train_test_split.split_train_test(churn_preproed_data)

    # model training


if __name__ == '__main__':
    data_path = "data/cell2celltrain.csv"
    main(data_path)
