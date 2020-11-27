import pandas as pd
import numpy as np
from itertools import combinations
from toolz import valmap



#Toy Example: Three stocks are available to buy IBM (IBM), Walmart (WMT), and Southern Electric (SEHI).
monthly_price = pd.read_csv("Toy_dataset.txt",sep = " ",header=None, names = ["IBM","WMT","SEHI"], index_col=0)

# Monthly return for each stock 
monthly_return = monthly_price.rolling(2).apply(lambda x: (x[1]-x[0])/x[0]).dropna()

# Average return (Expected return)
avg_return = list(monthly_return.mean())

#Covariance between each pair of stocks
co_variance = pd.DataFrame.cov(monthly_return)

# Risk aversion of the investor is set to 1, so the objective function will resemble the unconstrained one
risk_lambda = 1

# The total money to be invested in $ (Capital)
capital = 1000

# StDev of return for rach stock 
stdev = list(monthly_return.std())

# Sharpe_ration for each stock in an descending
Sharpe_ratio = [[stock, Return/Std] for Return, Std, stock in zip(avg_return, stdev, monthly_price.columns)]
Sharpe_ratio = sorted(Sharpe_ratio, key = lambda x: x[1], reverse=True)


# Greedy function:
def greedy(Sharpe_ratio, k, epsilon, delta, co_variance):
    Best_solution = {}
    V = 0
    for i in range(10000):
        # current solution
        S = {}
        # Random weights 
        for stock in range(k):
            S[Sharpe_ratio[stock][0]] = np.random.uniform(epsilon, delta)
        # Normalizing to sum up to 1
        S = valmap(lambda x: x/sum(S.values()), S)

        # All possible combinations of stock pairs in the current solution
        pair_comb = list(combinations(S.keys(), 2))

        # Objective function value for the current solution
        f = 0
        for comb in pair_comb:
            f += (float(co_variance[[comb[0]]].loc[comb[1]]) * (S[comb[0]]) * (S[comb[1]]))
        
        # update best solution
        if f > V :
            V = f
            Best_solution = S

    return Best_solution
                    


test = greedy(Sharpe_ratio, 3, 0.1, 1, co_variance)
print(test)