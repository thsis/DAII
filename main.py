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
X_train.shape
train.head(5)
test.shape
test.head(4)

X_test = test.loc[:, train.columns != "T2"].values
y_test = test.loc[:, "T2"].values.reshape((313, 1))

# Initialize Validation dataframe
Validate = test.loc[:, :]

# Single Hidden Layer Neural Nets.
nn1l_configs = glob(os.path.join("configs", "nn1l", "*.json"))

# DEBUG:
for config in nn1l_configs[: 2]:
    with open(config, 'r') as model_pars:
        params = json.load(model_pars)

    nn1l = ann.NN1L(data=X_train,
                    labels=y_train,
                    **params["init"])
    nn1l.train(**params["train"])
    validation = nn1l.predict(X_test)
    Validate[params["name"]] = validation.flatten()
    graphs.plot_decision_boundary(test.iloc[:, :29], "x6", "x13", "T2", nn1l,
                                  cmap=plt.cm.coolwarm, alpha=0.4)
    plt.show()


# Validate.to_csv(os.path.join("data", "validation.csv"), sep=';')
