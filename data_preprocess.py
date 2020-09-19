# -*- coding: utf-8 -*-
#!/user/bin/env python3
"""
@author:Xinlei Ma, Liye Peng
@description: preprocessing raw data
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.externals import joblib
import gc
from imblearn.over_sampling import SMOTE


base_dir = './'
base_model_dir = base_dir + 'models/'
base_result_dir = base_dir + 'result/'


def mulit_onehot_encoder(df, columns, isPredict):
    '''
    one-hot编码
    :param df:
    :param columns:
    :param isPredict: 是否是进行预测，如果是预测的话，直接使用模型，否则训练模型，并将模型结果保存成文件
    :return:
    '''
    if isPredict:
        for column_name in columns:
            # 加载encoder
            Enc_label = joblib.load(base_model_dir + column_name +
                                    ".label_encoder")
            Enc_ohe = joblib.load(base_model_dir + column_name +
                                  ".onehot_encoder")

            df['Dummies'] = Enc_label.transform(df[column_name])

            df_dummies = pd.DataFrame(Enc_ohe.transform(
                df[["Dummies"]]).todense(), columns=Enc_label.classes_)
            df_dummies.rename(columns=lambda x: column_name +
                                                "_" + x, inplace=True)  # 重新命名
            df = pd.concat([df, df_dummies], axis=1)
        df.drop(["Dummies"], axis=1, inplace=True)
        df.drop(columns, axis=1, inplace=True)
    else:
        Enc_ohe, Enc_label = OneHotEncoder(), LabelEncoder()
        for column_name in columns:
            Enc_label.fit(df[column_name])
            joblib.dump(Enc_label, base_model_dir + column_name +
                        ".label_encoder")
            df['Dummies'] = Enc_label.transform(df[column_name])
            Enc_ohe.fit(df[["Dummies"]])
            joblib.dump(Enc_ohe, base_model_dir + column_name +
                        ".onehot_encoder")

            df_dummies = pd.DataFrame(Enc_ohe.transform(
                df[["Dummies"]]).todense(), columns=Enc_label.classes_)
            df_dummies.rename(columns=lambda x: column_name +
                                                "_" + x, inplace=True)  # 重新命名
            df = pd.concat([df, df_dummies], axis=1)
        df.drop(["Dummies"], axis=1, inplace=True)
        df.drop(columns, axis=1, inplace=True)
    gc.collect()
    return df


def preprocess(churn_data_df):
    data_path = "./data/cell2celltrain.csv"
    churn_data_df = pd.read_csv(data_path)  # 读取数据为df
    churn_data_df.head()

    churn_data_df.fillna(method='ffill', inplace=True)

    # 计算相关系数，结果：无强相关性
    churn_data_df.corr()

    # 所有的值为Yes/No的列，都改成 0/1
    churn_data_df['Churn'] = churn_data_df['Churn'].map(dict(Yes=1, No=0))
    churn_data_df['ChildrenInHH'] = churn_data_df['ChildrenInHH'].map(dict(Yes=1, No=0))
    churn_data_df['HandsetRefurbished'] = churn_data_df['HandsetRefurbished'].map(dict(Yes=1, No=0))
    churn_data_df['HandsetWebCapable'] = churn_data_df['HandsetWebCapable'].map(dict(Yes=1, No=0))
    churn_data_df['TruckOwner'] = churn_data_df['TruckOwner'].map(dict(Yes=1, No=0))
    churn_data_df['RVOwner'] = churn_data_df['RVOwner'].map(dict(Yes=1, No=0))
    churn_data_df['Homeownership'] = churn_data_df['Homeownership'].map(dict(Known=1, Unknown=0))
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
    churn_data_df['MaritalStatus'] = churn_data_df['MaritalStatus'].map(dict(Yes=1, No=0, Unknown=-1))
    churn_data_df['HandsetPrice'] = churn_data_df['HandsetPrice'].replace('Unknown', 30)

    churn_data_df[['CreditRating_num', 'CreditRating_str']] = churn_data_df['CreditRating'].str.split('-', expand=True)
    churn_data_df['CreditRating'] = churn_data_df['CreditRating_num']
    churn_data_df['ServiceArea'] = churn_data_df['ServiceArea'].str[-3:]

    churn_data_df['PrizmCode'] = churn_data_df['PrizmCode'].map(dict(Other=0, Rural=1, Suburban=2, Town=3))
    churn_data_df['Occupation'] = churn_data_df['Occupation'].map(
        dict(Clerical=0, Crafts=1, Homemaker=2, Other=3, Professional=4, Retired=5, Self=6, Student=7))
    df_one_hot = churn_data_df[['PrizmCode', 'Occupation']].astype(str)
    df_one_hot = mulit_onehot_encoder(df_one_hot, ['PrizmCode', 'Occupation'], isPredict=0)

    churn_data_df.drop(['CreditRating_num', 'CreditRating_str'], axis=1, inplace=True)
    churn_data_df.drop(['PrizmCode', 'Occupation'], axis=1, inplace=True)
    churn_data_df = churn_data_df.astype(int)
    churn_data = churn_data_df.drop(['CustomerID', 'Churn'], axis=1, inplace=False)

    churn_data2 = churn_data.iloc[:, :54].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    df_result = pd.concat([churn_data2, df_one_hot], axis=1)
    df_result = pd.concat([churn_data_df['Churn'], df_result], axis=1)

    # 类别均衡处理
    y = df_result['Churn']
    X = df_result.drop(['Churn'], axis=1)
    # 建立模型
    smote_model = SMOTE()
    # 进行过抽样处理
    X_smote, y_smote = smote_model.fit_sample(X, y)
    # # 将特征值和目标值组合成一个DataFrame
    # smote_df = pd.concat([X_smote, y_smote], axis=1)

    # 切分测试集与训练集
    X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=.2, random_state=123)

    # print(np.array(X_train).shape)
    print(type(y_train))
    print(y_train)

    return X_train, X_test, y_train, y_test
