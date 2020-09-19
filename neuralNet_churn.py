# -*- coding: utf-8 -*-
#!/user/bin/env python3
"""
@author:Zhenhao Li
@description: NN churn prediction
"""

import model_validation


def NN_churn(X_train, y_train, X_test, y_test):

    y_pred = []

    accu_scr, prec_scr, rec_scr, f1_scr = model_validation.model_valid(y_test, y_pred)

    print("NN模型准确率 : " + str(accu_scr))
    print("NN模型精确率 : " + str(prec_scr))
    print("NN模型召回率 : " + str(rec_scr))
    print("NN模型F1 score : " + str(f1_scr))
