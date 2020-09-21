# -*- coding: utf-8 -*-
#!/user/bin/env python3
"""
@author:duyu
@description: LightGBM churn prediction
"""

import lightgbm as lgb
import model_validation
from sklearn.metrics import roc_curve, auc


def lightGBM_churn(X_train, y_train, X_test, y_test):
    lgb_model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=63, max_depth=-1, learning_rate=0.1,
                                   n_estimators=200, max_bin=255, subsample_for_bin=2000, objective=None,
                                   min_split_gain=0.0, min_child_weight=2, scale_pos_weight=2,
                                   min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                   reg_alpha=1, reg_lambda=1, random_state=12, n_jobs=8, silent=True)

    lgb_churn_model = lgb_model.fit(X_train, y_train, eval_metric='auc',
                                    eval_set=[(X_test, y_test)], early_stopping_rounds=100)
    y_pred = lgb_churn_model.predict(X_test)
    y_pred_proba = lgb_churn_model.predict_proba(X_test)
    accu_scr, prec_scr, rec_scr, f1_scr = model_validation.model_valid(y_test, y_pred)

    print("lightGBM模型准确率 : " + str(accu_scr))
    print("lightGBM模型精确率 : " + str(prec_scr))
    print("lightGBM模型召回率 : " + str(rec_scr))
    print("lightGBM模型F1 score : " + str(f1_scr))

    # generate roc
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc
