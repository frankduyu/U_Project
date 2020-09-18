# -*- coding: utf-8 -*-
#!/user/bin/env python3
"""
@author:duyu
@description: xgboost churn prediction
"""

import xgboost as xgb
import matplotlib.pyplot as plt
import xgboost_explainer as xgb_exp
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def xgboost_churn(train_data, train_label, test_data, test_label):
    # load conf
    name = "xgboost_churn"
    fea_path = "conf/u_plan_train_conf"
    col_names = ['y']

    with open(fea_path, 'r') as file:
        for line in file:
            line = line.strip('\n')
            col_names.append(line)
    real_fea_names = col_names[2:]

    # model training
    dtrain = xgb.DMatrix(train_data, train_label)
    lmda = 1.0
    params = {"objective": "binary:logistic", "max_depth": 6, "eta": 0.3, "gamma": 1,
              "colsample_bytree": 0.8, "min_child_weight": 2, "subsample": 0.8}
    best_iteration = 100
    bst = xgb.train(params, dtrain, best_iteration)

    # model validation
    dtest = xgb.DMatrix(test_data)
    y_pred = bst.predict(dtest)
    accu_scr = accuracy_score(test_label, y_pred)
    prec_scr = precision_score(test_label, y_pred)
    rec_scr = recall_score(test_label, y_pred)
    f1_scr = f1_score(test_label, y_pred)

    print("xgboost模型准确率 : " + str(accu_scr))
    print("xgboost模型精确率 : " + str(prec_scr))
    print("xgboost模型召回率 : " + str(rec_scr))
    print("xgboost模型F1 score : " + str(f1_scr))

    # get importance score and rank
    importance_score_dict = bst.get_fscore()
    importance_score_list = []
    for k, v in importance_score_dict.items():
        importance_score_list.append((v, int(k[1:])))
    importance_score_list = sorted(importance_score_list, key=lambda x: x[0], reverse=True)
    for i in range(len(importance_score_list)):
        print(real_fea_names[importance_score_list[i][1]] + '\t' + str(importance_score_list[i][0]))
    top_20_f = []
    for i in range(20):
        top_20_f.append(real_fea_names[importance_score_list[i][1]])

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
            if 'f'+str(idx) in dist:
                fea_logit[idx].append(-dist['f'+str(idx)])
            else:
                fea_logit[idx].append(0)

    # explainer plot
    fig = plt.figure(figsize=(30, 30))
    fig.suptitle('Feature User Churn')
    for i in range(len(top_20_f)):
        fea = top_20_f[i]
        idx = real_fea_names.index(fea)
        fea_data = train_data[fea].values
        ax = fig.add_subplot(5, 4, i+1)
        ax.set_xscale('log')
        ax.set_xlim(min([i for i in fea_data if i > 0]), max(fea_data))
        ax.scatter(fea_data, fea_logit[idx], s=0.1)
        ax.hlines(0, min([i for i in fea_data if i > 0]), max(fea_data), linewidth=0.5)
        ax.set_title(top_20_f[i])
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig('pic/' + name + '_' + str(best_iteration) + '.png', dpi=600)
