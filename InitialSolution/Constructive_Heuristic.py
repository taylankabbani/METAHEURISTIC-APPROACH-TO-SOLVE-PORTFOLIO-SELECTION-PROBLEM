import pandas as pd
import numpy as np
from itertools import combinations
from toolz import valmap
class GREEDY():
    '''
    Finds the initial solution based on greedy approach.
    Input: stock price dataframe, Risk aversion(Lambda), Capital($), assets in portfolio(k),upper and lower bounds (epsilon & delta), number of iterations(T)
    '''
    def __init__ (self, price_df, Lambda, Capital, k,epsilon, delta, T):
        self.price_df = price_df
        self.Lambda = Lambda
        self.capital = Capital
        self.k = k
        self.epsilon = epsilon
        self.delta = delta
        self.T = T
        self.return_df = self.get_return()
        self.avg_return = list(self.return_df.mean()) #Calculate avg return
        self.co_variance = pd.DataFrame.cov(self.return_df) #Covariance between each pair of stocks
        self.stdev = list(self.return_df.std())
        self.sharpe_ratio = self.get_sharpe_ratio()
        self.solution, self.obj_value = self.greedy_fun()

    # Calculate stock retun
    def get_return(self):
        return self.price_df.rolling(2).apply(lambda x: (x[1]-x[0])/x[0],raw=False).dropna()

    # Sharpe_ration for each stock in an descending order
    def get_sharpe_ratio(self):
        Sharpe_ratio = [[stock, Return/Std] for Return, Std, stock in zip(self.avg_return, self.stdev, self.price_df.columns)]
        Sharpe_ratio = sorted(Sharpe_ratio, key = lambda x: x[1], reverse=True)
        return Sharpe_ratio

    # Greedy function:
    def greedy_fun(self):
        Best_solution = {}
        V = float("inf")
        for i in range(self.T):
            # current solution
            S = {}
            #Random weights for k top stocks
            for stock in range(self.k):
                S[self.sharpe_ratio[stock][0]] = np.random.uniform(self.epsilon, self.delta)
            # Normalizing to sum up to 1
            S = valmap(lambda x: x/sum(S.values()), S)
            #All possible combinations of stock pairs in the current solution
            pair_comb = list(combinations(S.keys(), 2))
            # Objective function value for the current solution
            f = 0
            for comb in pair_comb:
                f += (float(self.co_variance[[comb[0]]].loc[comb[1]]) * (S[comb[0]]) * (S[comb[1]]))
            #update best solution
            if f < V :
                V = f
                Best_solution = S
        return Best_solution, V