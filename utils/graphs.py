"""
Utilities for plotting the decision boundary of a neural network.
Compare code with:
http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html

First: Find the two combination of ratios that looks the nicest.
Second: Create a numpy meshgrid & obtain predictions from the trained network.
Third: Plot the predictions by using a contour plot.
"""

import numpy as np
from matplotlib import pyplot as plt


def make_meshgrid(x, y, h=0.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def setup_contours(data, x, y, model):
    xpos = data.columns.get_loc(x)
    ypos = data.columns.get_loc(y)

    xx, yy = make_meshgrid(data[x].values,
                           data[y].values)

    xflat, yflat = xx.ravel(), yy.ravel()

    rep_data = np.ones((len(xflat), data.shape[1]-1))

    rep_data[:, xpos] = xflat
    rep_data[:, ypos] = yflat

    Z = model.predict(rep_data).reshape(xx.shape)
    return xx, yy, Z


def plot_decision_boundary(data, x, y, labels, model, **kwargs):
    xx, yy, Z = setup_contours(data=data, x=x, y=y, model=model)

    x0, x1 = data[x].values, data[y].values
    x0lim = x0.min(), x0.max()
    x1lim = x1.min(), x1.max()

    col = data[labels].values

    plt.scatter(x0, x1, c=col, **kwargs)
    plt.contourf(xx, yy, Z, **kwargs)
    plt.xlim(x0lim)
    plt.ylim(x1lim)
    plt.xlabel(x)
    plt.ylabel(y)
