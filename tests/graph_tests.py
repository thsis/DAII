import os
import pandas as pd
import numpy as np
# import seaborn as sns
from matplotlib import pyplot as plt
from utils import graphs, roc
from models import ann

np.random.seed(42)

# Set up test data
data_path = os.path.join("data", "ratios.csv")
ratios = pd.read_csv(data_path, sep=';')
ratios.head(5)

# Take all insolvent and a sample of solvent firms.
insolvent = ratios.loc[ratios['T2'] == 1, :]
solvent = ratios.loc[ratios['T2'] == 0, :]

sample = pd.concat([insolvent,
                    solvent.sample(insolvent.shape[0])])
sample.shape
sample.head(3)

# That's an awfully big plot...
# plt.savefig("tests/pairplot.pdf")
# sns.pairplot(sample, hue='T2', markers=['o', 's'])

var = ['x'+str(i) for i in range(6, 21)] + ['T2']
subset = sample.loc[:, var]
subset.head(3)
# sns.pairplot(subset, hue='T2', markers=['o', 's'])
type(subset)
subset.shape
# plt.show()

X = subset.loc[:, subset.columns != "T2"].values
y = subset.loc[:, "T2"].values.reshape((1564, 1))

nn1l = ann.NN1L(data=X, labels=y, units=10)
nn1l.train(epochs=1000, learn=0.5)


graphs.plot_decision_boundary(subset, "x6", "x13", "T2", nn1l,
                              cmap=plt.cm.coolwarm, alpha=0.4)

plt.show()

# TRY:
# x17 vs x18
