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
