# DAII

## Making this work
### On Linux & Mac
1. Clone the repository.
2. Open a terminal and change the directory: `cd <your_cloned_repository>`
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

3. Implement feed-forward-neural-network.
    - __NN1L__ (defined here): function class for neural net with one hidden layer.  

4. Perform grid-search.
