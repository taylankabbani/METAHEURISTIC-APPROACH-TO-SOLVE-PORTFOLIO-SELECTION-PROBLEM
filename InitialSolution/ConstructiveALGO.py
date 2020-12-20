import pandas as pd
from itertools import combinations

class GREEDY():
    '''
    Creates an initial solution based on a greedy approach.
    Input: mean_return & SD filePath, Correlation filePath, Risk aversion(Lambda), Capital($), assets in portfolio(k),upper and lower bounds (epsilon & delta), number of iterations(T)
    '''
    def __init__ (self, ReturnSD_path, corr_path, Lambda, Capital, k,epsilon, delta, T):
        # self.ReturnSD_dict =  pd.read_csv(ReturnSD_path,sep = " ",header=None, names = ["Mean_return","SD"], index_col = 0)
        self.ReturnSD, self.corr = self.get_data(ReturnSD_path, corr_path)
        self.Lambda = Lambda
        self.capital = Capital
        self.k = k
        self.epsilon = epsilon
        self.delta = delta
        self.T = T


    def get_data(self, ReturnSD_path, corr_path):
        ReturnSD_dict = dict()
        corr_dict = dict()
        with open(corr_path) as fp:
            for line in fp:
                key = (int(line.split(" ")[0]), int(line.split(" ")[1]))
                value = float(line.split(" ")[2].strip('\n'))
                corr_dict[key] = value
        with open(ReturnSD_path) as fp:
            for line in fp:
                key = line.split(" ")[0]
                value = {'Mean_Return': float(line.split(" ")[1]), 'SD': float(line.split(" ")[2].strip('\n'))}
                ReturnSD_dict[key] = value
        return ReturnSD_dict, corr_dict

    # Sharpe_ration for each stock in an descending order
    def get_sharpe_ratio(self):
        Sharpe_ratio = [[key, value['Mean_Return']/value['SD']] for key, value in self.ReturnSD.items()]
        Sharpe_ratio = sorted(Sharpe_ratio, key = lambda x: x[1], reverse=True)
        return Sharpe_ratio

    def Objfun(self, solution):
        '''Takes a dict as solution (stock:weight).
        returns the multi objective function value of the solution
        '''
        N = len(solution) #number of stocks in portfolio 
        # Objective 1 value for the current solution (sum over all assets in the portfolio)
        objvalue_1 = 0
        for i in range(1,N+1):
            for j in range(1,N+1):
                print(solution[str(i)] * solution[str(j)], self.corr[(j,i)])
            # f += (float(self.co_variance[[comb[0]]].loc[comb[1]]) * (S[comb[0]]) * (S[comb[1]]))
        # print(pair_comb)


    # Greedy function:
    def greedy_fun(self):
        Best_solution = {}
        V = float("inf")
        sharp_ration = self.get_sharpe_ratio
        for i in range(self.T):
            # current solution
            S = {}
            #Random weights for k top stocks
            for stock in range(self.k):
                S[sharpe_ratio[stock][0]] = np.random.uniform(self.epsilon, self.delta)
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

test = GREEDY(ReturnSD_path= "Data/Hong_kong_31/Return&SD.txt", corr_path="Data/Hong_kong_31/correlation.txt", Lambda=0, Capital= 1000, k= 3, epsilon=0.1, delta=1, T=1000)
# print(test.corr[(1,2)])
solution = {'1': 0.442, '2' : 0.1036, '3':0.454}
test.Objfun(solution)
# for key, value in test.ReturnSD.items():
#     Sharpe_ratio.append([key, value['Mean_Return']/value['SD']])

# Sharpe_ratio
# # Sharpe_ratio = [[key, Return/Std] for Return, Std, stock in zip(self.avg_return, self.stdev, self.price_df.columns)]
