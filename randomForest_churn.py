# -*- coding: utf-8 -*-
#!/user/bin/env python3
"""
@author:Zhe Han
@description: random forest churn prediction
"""
from sklearn.ensemble import RandomForestClassifier
import model_validation
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def random_forest_churn(X_train, y_train, X_test, y_test):
    fea_path = "conf/u_plan_train_conf"
    col_names = []

    with open(fea_path, 'r') as file:
        for line in file:
            line = line.strip('\n')
            col_names.append(line)

    # model construction
    estimator_RF = RandomForestClassifier(n_estimators=100, random_state=9, n_jobs=-1)

    # model training
    model_RF = estimator_RF.fit(X_train, y_train)

    # model prediction
    predict_RF = model_RF.predict(X_test)
    y_predict_proba = model_RF.predict_proba(X_test)
    model_validation.model_valid(y_test, predict_RF)
    accu_scr, prec_scr, rec_scr, f1_scr = model_validation.model_valid(y_test, predict_RF)

    print("RandomForest模型准确率 : " + str(accu_scr))
    print("RandomForest模型精确率 : " + str(prec_scr))
    print("RandomForest模型召回率 : " + str(rec_scr))
    print("RandomForest模型F1 score : " + str(f1_scr))

    # generate roc
    fpr, tpr, thresholds = roc_curve(y_test, y_predict_proba[:, 1])
    roc_auc = auc(fpr, tpr)

    # get RF feature importance
    importance_score_list = []
    for i in range(len(model_RF.feature_importances_)):
        importance_score_list.append([model_RF.feature_importances_[i], col_names[i]])
    importance_score_list = sorted(importance_score_list, key=lambda x: x[0], reverse=True)
    # get RF importance features
    print("=" * 50)
    print("xgboost feature importance")
    for i in range(len(importance_score_list)):
        print(importance_score_list[i][1] + '\t' + str(importance_score_list[i][0]))
    print("=" * 50)

    # # 得分
    # score_RF_train = model_RF.score(X_train, y_train)
    # score_RF_test = model_RF.score(X_test, y_test)

    # # 特征重要性
    # fig, ax = plt.subplots(figsize=(7, 5))
    # ax.bar(col_names, model_RF.feature_importances_)
    # ax.set_title("Feature Importances")
    # fig.savefig('pic/RF_feature_importance.png')

    return fpr, tpr, roc_auc
