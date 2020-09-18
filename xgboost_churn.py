# -*- coding: gb18030 -*-
#!/user/bin/env python

import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import xgboost_explainer as xgb_exp
import pickle


# def xgboost_churn(train_data, test_data):
def xgboost_churn():
    name = "xgboost_churn"
    fea_path = "conf/u_plan_train_conf"
    col_names = ['y']

    with open(fea_path, 'r') as file:
        for line in file:
            line = line.strip('\n')
            col_names.append(line)

    real_fea_names = col_names[2:]

    train_data = pd.read_table("data/test_train_data.txt", header=None, index_col=None)
    train_data.columns = col_names
    train_label = train_data['y']
    train_data = train_data.drop(['y'], axis=1)
    train_label = train_label.values
    train_label = [0 if int(label) == 0 else 1 for label in train_label]

    dtrain = xgb.DMatrix(train_data.values, train_label)
    lmda = 1.0

    params = {"objective": "binary:logistic", "max_depth": 3, "eta": 0.1, "gamma": 1,
              "colsample_bytree": 0.8, "min_child_weight": 2, "subsample": 1}
    best_iteration = 30
    watchlist = [(dtrain, 'train')]
    bst = xgb.train(params, dtrain, best_iteration, watchlist)

    # get importance score and rank
    importance_score_dict = bst.get_fscore()
    importance_score_list = []
    for k, v in importance_score_dict.items():
        importance_score_list.append((v, int(k[1:])))
    importance_score_list = sorted(importance_score_list, key=lambda x: x[0], reverse=True)
    for i in range(len(importance_score_list)):
        print(real_fea_names[importance_score_list[i][1]] + '\t' + str(importance_score_list[i][0]))

    top_5_f = []
    for i in range(5):
        top_5_f.append(real_fea_names[importance_score_list[i][1]])

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

    fig = plt.figure(figsize=(30, 30))
    fig.suptitle('Feature User Churn')
    for i in range(len(top_5_f)):
        fea = top_5_f[i]
        idx = real_fea_names.index(fea)
        fea_data = train_data[fea].values
        ax = fig.add_subplot(3, 2, i+1)
        ax.set_xscale('log')
        ax.set_xlim(min([i for i in fea_data if i > 0]), max(fea_data))
        ax.scatter(fea_data, fea_logit[idx], s=0.1)
        ax.hlines(0, min([i for i in fea_data if i > 0]), max(fea_data), linewidth=0.5)
        ax.set_title(top_5_f[i])
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig('pic/' + name + '_' + str(best_iteration) + '.png', dpi=800)

