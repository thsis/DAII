# DAII

## Running the code from this repository
### On Linux & Mac
1. Clone the repository.
2. Open a terminal and change the directory: `cd <path_to_your_cloned_repository>`
3. run ```pip install -r requirements.txt```

### On Windows
1. Install git for Windows [from here](https://gitforwindows.org/). This ensures that you have access to *BASH*. Open git-Bash and follow the steps for Linux and Mac
2. Alternatively you can use the Windows-command-line performing the same steps.

## Guideline through the repository
1. Preprocess data: clean data and calculate ratios.
    - __credit_clean.csv__: contains the cleaned dataset from Creditreform.
    - __ratios.csv__: contains only the 28 finantial ratios.
    - __full.csv__: contains all 28 finantial ratios as well as the original features.

2. Perform exploratory factor analysis.
    - __EFA.R__ [defined here](https://github.com/thsis/DAII/blob/master/factor_analysis/EFA.R): script containing code for exploratory factor analysis.

3. Implement feed-forward-neural-network.
    - __NN1L__ [defined here](https://github.com/thsis/DAII/blob/master/models/ann.py): function class for neural net with one hidden layer.
    - __NN2L__ [defined here](https://github.com/thsis/DAII/blob/master/models/ann.py): function class
    for neual net with two hidden layers.

4. Perform grid-search.
