# DAII
## Introduction

This repository accompanies my submission to the course Data-Analysis II at the Humboldt university in Berlin. However, during my research I found it rather difficult to find examples of feed forward neural networks which contained more than one hidden layer. So if you aren't my lecturer by chance, you probably have the same problem. For you my fellow machine-learning starter, I will try to explain what I did. In the *models* directory you will find `tensorflow` implementations for simple neural networks, neural networks with two and neural networks with five hidden layers.

If you are my lecturer I have to clarify that I don't consider the README to be part of my submission.

## Running the code from this repository
### On Linux & Mac
1. Clone the repository.
2. Open a terminal and change the directory: `cd <path_to_your_cloned_repository>`
3. run ```pip install -r requirements.txt```

### On Windows
1. Install git for Windows [from here](https://gitforwindows.org/). This ensures that you have access to *BASH*. Open git-Bash and follow the steps for Linux and Mac
2. Alternatively you can use the Windows-command-line performing the same steps.

## How to use the neural network architectures
### Network with one hidden layer

For a simple neural network with only one hidden layer you can use the `NN1L` class. First, create an instance then you can train the network. Assuming you have your features stored in a `numpy.ndarray` named `X` and your labels stored as a column vector `y` (i.e. `y.shape`=`(n, 1)`), where $n$ is equal to the number of observations. You can set up a neural network with 10 neurons by running:

```
from models import ann

nn1l = ann.NN1L(data=X, labels=y, units=10)
```

Training is done by the `train` method. You can control the training-process through the parameters:

* epochs: Number of training iterations.
* learn: Learning rate of the gradient descent optimizer.

The `tensorboard` logfiles are stored in `logdir`, it defaults to the current directory. Say you want to train for 100.000 epochs with a learning rate of 0.07

```
nn1l.train(epochs=100000, learn=0.07)
```

Predictions can be obtained by running the `predict` method. You only need to specify one parameter `data`, which expects a `numpy.ndarray` whose number of columns matches your initial data.

```
nn1l.predict(X_test)
```

### Network with multiple hidden layers

If you are interested in training networks with multiple hidden layers, two and five hidden layers in particular, you can use the `NN2L` and `NN5L`-classes respectively. The only difference to `NN1L`-instances lies in the way they are created. The `units`-parameter expects a sequence that describes the architecture of your network. The first element of the sequence is reserved for the input layer. For a 2-hidden-layer network you need a list or a tuple of length 3. For neural network with five hidden layers you need a sequence of length six. If you have a `ǹumpy.ndarray` `X` with 50 features and want to train a two-layer network with 10 units in the first and 4 units in the second hidden layer you can do that with:

```
nn2l = ann.NN2L(data=X, labels=y, units=(50, 10, 4))
```

Say you want to implement a five-layer architecture that starts with 10 units and gradually removes one neuron from each layer until ends with 6. On the same dataset `X` as before you can try:

```
nn5l = ann.NN5L(data=X, labels=y, units=(50, 10, 9, 8, 7, 6))
```

The methods for training and predicting are the same as in `NN1L` instances.

## Guideline through the repository
This section outlines my train of thought. I believe this order is the best way of understanding what is going on.

1. Preprocess data: clean data and calculate ratios.
    - __credit_clean.csv__: contains the cleaned dataset from Creditreform.
    - __ratios.csv__: contains only the 28 finantial ratios.
    - __full.csv__: contains all 28 finantial ratios as well as the original features.

2. Implement feed-forward-neural-network.
    - __NN1L__ [defined here](https://github.com/thsis/DAII/blob/master/models/ann.py): function class for neural net with one hidden layer.
    - __NN2L__ [defined here](https://github.com/thsis/DAII/blob/master/models/ann.py): function class
    for neural net with two hidden layers.
    - __NN5L__ [defined here](https://github.com/thsis/DAII/blob/master/models/ann.py): function class
    for neural net with five hidden layers (which for most applications is probably overkill).

3. Plot Roc curves and contour plots for the previously implemented neural networks.
    - __graphs.py__ [defined here](https://github.com/thsis/DAII/blob/master/utils/graphs.py): 2D-contour plots of the neural network's output.
    - __roc.py__ [defined here](https://github.com/thsis/DAII/blob/master/utils/roc.py): Receiver Operator Characteristic for the neural network's predictions.

4. Create configuration files that can be looped over.
    - __generate_configs.py__ [defined here](https://github.com/thsis/DAII/blob/master/utils/generate_configs.py): an elegant way to train multiple models is to unpack the key-value-pairs of a dictionary. For this one can use the syntax `foo(**dict)`.

5. Perform grid-search.
    - __main.py__ [defined here](https://github.com/thsis/DAII/blob/master/main.py): Contains the training loop and plots the figures.
