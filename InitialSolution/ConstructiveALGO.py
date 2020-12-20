class GREEDY():
    '''
    Creates an initial solution based on a greedy approach.
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