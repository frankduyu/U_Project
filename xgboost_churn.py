# -*- coding: utf-8 -*-
#!/user/bin/env python3
"""
@author:duyu
@description: xgboost churn prediction
"""

import xgboost as xgb
import matplotlib.pyplot as plt
import xgboost_explainer as xgb_exp
import model_validation
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings("ignore")


def xgboost_churn(X_train, y_train, X_test, y_test):
    # load conf
    name = "xgboost_churn"
    fea_path = "conf/u_plan_train_conf"
    col_names = ['y']

    with open(fea_path, 'r') as file:
        for line in file:
            line = line.strip('\n')
            col_names.append(line)
    real_fea_names = col_names[1:]

    X_df = pd.DataFrame(X_train, columns=real_fea_names)

    # model training
    dtrain = xgb.DMatrix(X_train, y_train)
    lmda = 1.0
    params = {"objective": "binary:logistic", "max_depth": 6, "eta": 0.3, "gamma": 1,
              "colsample_bytree": 0.8, "min_child_weight": 2, "subsample": 0.8}
    best_iteration = 150
    bst = xgb.train(params, dtrain, best_iteration)

    # model validation
    dtest = xgb.DMatrix(X_test)
    y_pred_proba = bst.predict(dtest)
    y_pred = np.array([1 if item >= 0.5 else 0 for item in y_pred_proba])
    accu_scr, prec_scr, rec_scr, f1_scr = model_validation.model_valid(y_test, y_pred)

    print("xgboost模型准确率 : " + str(accu_scr))
    print("xgboost模型精确率 : " + str(prec_scr))
    print("xgboost模型召回率 : " + str(rec_scr))
    print("xgboost模型F1 score : " + str(f1_scr))

    # generate roc
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # get importance score and rank
    importance_score_dict = bst.get_score(importance_type="gain")
    importance_score_list = []
    for k, v in importance_score_dict.items():
        importance_score_list.append((v, int(k[1:])))
    importance_score_list = sorted(importance_score_list, key=lambda x: x[0], reverse=True)
    # get xgboost importance features
    print("=" * 50)
    print("xgboost feature importance")
    for i in range(len(importance_score_list)):
        print(real_fea_names[importance_score_list[i][1]] + '\t' + str(importance_score_list[i][0]))
    print("=" * 50)
    top_12_f = []
    for i in range(11):
        top_12_f.append(real_fea_names[importance_score_list[i][1]])
    top_12_f.append('CustomerCareCalls')

    # explainer model training
    params = {"objective": "binary:logistic", 'silent': 1, 'eval_metric': 'auc', 'base_score': 0.5, "lambda": lmda}
    bst = xgb.train(params, dtrain, best_iteration)

    # calculate xgboost probability
    tree_lst = xgb_exp.model2table(bst, lmda=lmda)
    leaf_lsts = bst.predict(dtrain, pred_leaf=True)
    fea_logit = [[] for _ in range(len(real_fea_names))]
    for i, leaf_lst in enumerate(leaf_lsts):
        dist = xgb_exp.logit_contribution(tree_lst, leaf_lst)
        for idx in range(len(real_fea_names)):
            if 'f' + str(idx) in dist:
                fea_logit[idx].append(-dist['f' + str(idx)])
            else:
                fea_logit[idx].append(0)

    # explainer plot
    fig = plt.figure(figsize=(22, 20))
    fig.suptitle('Feature User Churn')
    for i in range(len(top_12_f)):
        fea = top_12_f[i]
        idx = real_fea_names.index(fea)
        fea_data = X_df[fea].values
        ax = fig.add_subplot(4, 3, i + 1)
        # ax.set_xscale('log')
        ax.set_xlim(min([i for i in fea_data if i > 0]), max(fea_data))
        ax.scatter(fea_data, fea_logit[idx], s=0.1)
        ax.hlines(0, min([i for i in fea_data if i > 0]), max(fea_data), linewidth=0.5)
        ax.set_title(top_12_f[i])
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig('pic/' + name + '_' + str(best_iteration) + '_v3.png', dpi=800)

    return fpr, tpr, roc_auc
