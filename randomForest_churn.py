# -*- coding: utf-8 -*-
#!/user/bin/env python3
"""
@author:Zhe Han
@description: random forest churn prediction
"""
from sklearn.ensemble import RandomForestClassifier
import model_validation


def random_forest_churn(X_train, y_train, X_test, y_test):
    # 构建模型
    estimator_RF = RandomForestClassifier(n_estimators=100, random_state=9, n_jobs=-1)
    # 拟合
    model_RF = estimator_RF.fit(X_train, y_train)

    # 预测
    predict_RF = model_RF.predict(X_test)
    model_validation.model_valid(y_test, predict_RF)
    accu_scr, prec_scr, rec_scr, f1_scr = model_validation.model_valid(y_test, predict_RF)

    print("RandomForest模型准确率 : " + str(accu_scr))
    print("RandomForest模型精确率 : " + str(prec_scr))
    print("RandomForest模型召回率 : " + str(rec_scr))
    print("RandomForest模型F1 score : " + str(f1_scr))

    # # 得分
    # score_RF_train = model_RF.score(X_train, y_train)
    # score_RF_test = model_RF.score(X_test, y_test)

    # # 特征重要性
    # fig, ax = plt.subplots(figsize=(7, 5))
    # ax.bar(range(len(model_RF.feature_importances_)), model_RF.feature_importances_)
    # ax.set_title("Feature Importances")
    # fig.show()
