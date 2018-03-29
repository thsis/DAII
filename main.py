"""Main script for analysis."""
import os
import json
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from glob import glob
from models import ann
from utils import graphs, roc
from matplotlib import pyplot as plt

np.random.seed(42)

# Set up train data
data_path = os.path.join("data", "ratios.csv")
ratios = pd.read_csv(data_path, sep=';', index_col=False)

insolvent = ratios.loc[ratios['T2'] == 1, :]
solvent = ratios.loc[ratios['T2'] == 0, :]

data = pd.concat([insolvent, solvent.sample(insolvent.shape[0])])

train, test = train_test_split(data, test_size=0.2, stratify=data["T2"])
X_train = train.loc[:, train.columns != "T2"].values
y_train = train.loc[:, "T2"].values.reshape((1251, 1))

X_test = test.loc[:, train.columns != "T2"].values
y_test = test.loc[:, "T2"].values.reshape((313, 1))

# Initialize Validation dataframe
Validate = test.loc[:, :]

# Set up Variables for contour-plots.
plot_x = ["x" + str(i) for i in range(1, 14)]
plot_y = ["x" + str(i) for i in range(14, 27)]
plot_vars = zip(plot_x, plot_y)

print("Single Hidden Layer Neural Nets.")
nn1l_configs = glob(os.path.join("configs", "nn1l", "*.json"))

for config in nn1l_configs:
    with open(config, 'r') as model_pars:
        params = json.load(model_pars)

    nn1l = ann.NN1L(data=X_train,
                    labels=y_train,
                    **params["init"])
    nn1l.train(**params["train"])
    validation = nn1l.predict(X_test)
    Validate[params["name"]] = validation.flatten()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    for i, (x, y) in enumerate(plot_vars):
        fig_path = os.path.join(params["train"]["logdir"],
                                params["name"] + "_" + str(i) + "_.png")
        graphs.plot_decision_boundary(test.iloc[:, :29], x, y, "T2", nn1l,
                                      cmap=plt.cm.coolwarm, alpha=0.4)
        plt.savefig(fig_path)
        plt.clf()

print("Two Hidden Layers Neural Nets.")
nn2l_configs = glob(os.path.join("configs", "nn2l", "*.json"))

for config in nn2l_configs:
    with open(config, 'r') as model_pars:
        params = json.load(model_pars)

    nn2l = ann.NN2L(data=X_train,
                    labels=y_train,
                    **params["init"])
    nn2l.train(**params["train"])
    validation = nn2l.predict(X_test)
    Validate[params["name"]] = validation.flatten()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    for i, (x, y) in enumerate(plot_vars):
        fig_path = os.path.join(params["train"]["logdir"],
                                params["name"] + "_" + str(i) + "_.png")
        graphs.plot_decision_boundary(test.iloc[:, :29], x, y, "T2", nn2l,
                                      cmap=plt.cm.coolwarm, alpha=0.4)
        plt.savefig(fig_path)
        plt.clf()

print("Five Hidden Layers Neural Nets.")
nn5l_configs = glob(os.path.join("configs", "nn5l", "*.json"))

for config in nn5l_configs:
    with open(config, 'r') as model_pars:
        params = json.load(model_pars)

    nn5l = ann.NN5L(data=X_train,
                    labels=y_train,
                    **params["init"])
    nn5l.train(**params["train"])
    validation = nn5l.predict(X_test)
    Validate[params["name"]] = validation.flatten()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    for i, (x, y) in enumerate(plot_vars):
        fig_path = os.path.join(params["train"]["logdir"],
                                params["name"] + "_" + str(i) + "_.png")
        graphs.plot_decision_boundary(test.iloc[:, :29], x, y, "T2", nn5l,
                                      cmap=plt.cm.coolwarm, alpha=0.4)
        plt.savefig(fig_path)
        plt.clf()


Validate.to_csv(os.path.join("data", "validation.csv"), sep=';')

# Roc for one Hidden Layer
fpr_1l, tpr_1l, auc_1l = roc.roc(preds=Validate.iloc[:, 29:41].values,
                                 labels=Validate.iloc[:, 28].values,
                                 names=list(Validate.iloc[:, 29:41].columns))
plt.savefig("rocNN1L.png")
plt.show()
plt.clf()
# Roc for two Hidden Layers
fpr_2l, tpr_2l, auc_2l = roc.roc(preds=Validate.iloc[:, 41:53].values,
                                 labels=Validate.iloc[:, 28].values,
                                 names=list(Validate.iloc[:, 41:53].columns))
plt.savefig("rocNN2L.png")
plt.show()
plt.clf()

# Roc for five Hidden Layers
fpr_5l, tpr_5l, auc_5l = roc.roc(preds=Validate.iloc[:, 54:].values,
                                 labels=Validate.iloc[:, 28].values,
                                 names=list(Validate.iloc[:, 54:].columns))
plt.savefig("rocNN5L.png")
plt.show()
plt.clf()
