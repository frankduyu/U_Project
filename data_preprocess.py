# -*- coding: utf-8 -*-
#!/user/bin/env python3
"""
@author:Xinlei Ma, Liye Peng
@description: preprocessing raw data
"""
import pandas as pd
import numpy as np


def preprocess(churn_data_df):
    # 所有的值为Yes/No的列，都改成 0/1
    churn_data_df['Churn'] = churn_data_df['Churn'].map(dict(Yes=1, No=0))
    churn_data_df['ChildrenInHH'] = churn_data_df['ChildrenInHH'].map(dict(Yes=1, No=0))
    churn_data_df['HandsetRefurbished'] = churn_data_df['HandsetRefurbished'].map(dict(Yes=1, No=0))
    churn_data_df['HandsetWebCapable'] = churn_data_df['HandsetWebCapable'].map(dict(Yes=1, No=0))
    churn_data_df['TruckOwner'] = churn_data_df['TruckOwner'].map(dict(Yes=1, No=0))
    churn_data_df['RVOwner'] = churn_data_df['RVOwner'].map(dict(Yes=1, No=0))
    churn_data_df['BuysViaMailOrder'] = churn_data_df['BuysViaMailOrder'].map(dict(Yes=1, No=0))
    churn_data_df['RespondsToMailOffers'] = churn_data_df['RespondsToMailOffers'].map(dict(Yes=1, No=0))
    churn_data_df['OptOutMailings'] = churn_data_df['OptOutMailings'].map(dict(Yes=1, No=0))
    churn_data_df['NonUSTravel'] = churn_data_df['NonUSTravel'].map(dict(Yes=1, No=0))
    churn_data_df['OwnsComputer'] = churn_data_df['OwnsComputer'].map(dict(Yes=1, No=0))
    churn_data_df['HasCreditCard'] = churn_data_df['HasCreditCard'].map(dict(Yes=1, No=0))
    churn_data_df['NewCellphoneUser'] = churn_data_df['NewCellphoneUser'].map(dict(Yes=1, No=0))
    churn_data_df['NotNewCellphoneUser'] = churn_data_df['NotNewCellphoneUser'].map(dict(Yes=1, No=0))
    churn_data_df['OwnsMotorcycle'] = churn_data_df['OwnsMotorcycle'].map(dict(Yes=1, No=0))
    churn_data_df['MadeCallToRetentionTeam'] = churn_data_df['MadeCallToRetentionTeam'].map(dict(Yes=1, No=0))
    churn_data_df['MaritalStatus'] = churn_data_df['MaritalStatus'].map(dict(Yes=1, No=0))

    # 将CreditRating这一列处理成只有数值的
    churn_data_df[['CreditRating_num', 'CreditRating_str']] = churn_data_df['CreditRating'].str.split('-', expand=True)

    # 删除 customerid 列
    churn_data_df.drop('CustomerID', axis=1, inplace=True)
    # 删除 creditRating_str & creditRating 列
    churn_data_df.drop(['CreditRating_str', 'CreditRating'], axis=1, inplace=True)

    # 将 credictRating_num 列转换为数值型
    churn_data_df[['CreditRating_num']] = churn_data_df[['CreditRating_num']].apply(pd.to_numeric)

    # # 看有Unknown的列及其占比
    # for i in churn_data_df.keys():
    #     t = round(len([x for x in churn_data_df[i] if x == 'Unknown']) / churn_data_df.shape[0], 3)
    #     if t > 0:
    #         print(i + ": ", t)
    #
    # # HandsetPrice 列的组成部分
    # print(churn_data_df['HandsetPrice'].value_counts())
    #
    # # Homeownership 列的组成部分
    # print(churn_data_df['Homeownership'].value_counts())
    #
    # # 每列 missing value 的个数
    # for i in churn_data_df.keys():
    #     t = sum([x for x in churn_data_df[i].isna() if x == True])
    #     if t > 0:
    #         print(i + ': ', t)
    #
    # # 计算所有特征与标签的相关系数，如强相关则删除该列
    # num_features = churn_data_df.select_dtypes(include=[np.number])
    # corr = num_features.corr()
    # print(corr)


