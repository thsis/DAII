"""
Define function that calculates and plots ROC-Metrics/Curves.

Compare with:
http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
"""

import numpy as np
from sklearn import metrics
from matplotlib import pyplot as plt


fake_preds = np.random.uniform(size=(100, 3))
fake_labels = (fake_preds[:, 2] + np.random.uniform(0, 0.5, 100)).round()


def roc(preds, labels, names=None):
    """
    Calculate ROC-metrics and plot curve for binary classifiers.
    --------------------------------------
    Parameters:
        - preds: 2D array of predictions. Values should be inside [0 ,1].
        - labels: 1D array of labels.
    Calculate True Positive Rate (tpr), False Positive Rate (fpr) and Area
    under curve (AUC).
    """

    _, m = preds.shape
    if not names:
        names = range(1, m+1)

    fpr = dict()
    tpr = dict()
    auc = dict()
    for i in range(m):
        fpr[i], tpr[i], _ = metrics.roc_curve(labels, preds[:, i])
        auc[i] = metrics.auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i],
                 label='model {name}, AUC={auc:.2}'.format(name=names[i],
                                                           auc=auc[i]), lw=2)

    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operator Characteristic")
    plt.legend(loc="lower right")
    plt.grid()
    return fpr, tpr, auc
