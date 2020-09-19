# -*- coding: utf-8 -*-
#!/user/bin/env python3
"""
@author:duyu
@description: validate model prediction
"""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def model_valid(y_true, y_pred):
    accu_scr = accuracy_score(y_true, y_pred)
    prec_scr = precision_score(y_true, y_pred)
    rec_scr = recall_score(y_true, y_pred)
    f1_scr = f1_score(y_true, y_pred)

    # conf_mat = confusion_matrix(y_true, y_pred)
    # prec_scr = conf_mat[0][0] / (conf_mat[0][0] + conf_mat[1][0])
    # rec_scr = conf_mat[0][0] / (conf_mat[0][0] + conf_mat[0][1])
    # f1_scr = 2 * conf_mat[0][0] / (2 * conf_mat[0][0] + conf_mat[1][0] + conf_mat[0][1])

    return accu_scr, prec_scr, rec_scr, f1_scr


