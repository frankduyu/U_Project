# -*- coding: utf-8 -*-
#!/user/bin/env python3
"""
@author:duyu
@description: Draw model ROC Curve
"""

import matplotlib.pyplot as plt


def drawRoc(xgb_params, gbm_params, rf_params, dnn_params):
    plt.subplots(figsize=(7, 5.5))
    plt.plot(xgb_params[0], xgb_params[1], color='darkorange', lw=2,
             label='xgb ROC curve (area = %0.2f)' % xgb_params[2])
    plt.plot(gbm_params[0], gbm_params[1], color='cyan', lw=2,
             label='lightGBM ROC curve (area = %0.2f)' % gbm_params[2])
    plt.plot(rf_params[0], rf_params[1], color='red', lw=2, label='RF ROC curve (area = %0.2f)' % rf_params[2])
    plt.plot(dnn_params[0], dnn_params[1], color='green', lw=2,
             label='DNN ROC curve (area = %0.2f)' % dnn_params[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('pic/roc.png')
