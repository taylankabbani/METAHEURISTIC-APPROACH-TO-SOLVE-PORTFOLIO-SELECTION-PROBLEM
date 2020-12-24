from toolz import valmap
import numpy as np
from InitialSolution import ConstructiveALGO as CH

class TS_ShortMemory():
    '''
    Input: mean_return & SD filePath, Correlation filePath, Risk aversion(Lambda), assets in portfolio(k),
    upper and lower bounds (epsilon & delta)
    '''
    def __init__(self, ReturnSD_path, corr_path, Lambda, k, epsilon, delta):
        self.ReturnSD, self.corr = self.get_data(ReturnSD_path, corr_path)
        self.cov = self.get_cov_dict()
        self.Lambda = Lambda
        self.k = k
        self.epsilon = epsilon
        self.delta = delta
        self.initial_solution, self.initial_objvalue = self.get_initial_solution()
        self.tabu_str = self.get_tabuestructure()



    def get_data(self, ReturnSD_path, corr_path):
        '''Returns dict of ReturnSD[stock: {Mean_Return, StandardDeviation}] and a dict of corr[(stocki,stockj:corr_value)]
        '''
        ReturnSD_dict = dict()
        corr_dict = dict()
        with open(corr_path) as fp:
            for line in fp:
                key = (int(line.split(" ")[0]), int(line.split(" ")[1]))
                value = float(line.split(" ")[2].strip('\n'))
                corr_dict[key] = value
        with open(ReturnSD_path) as fp:
            for line in fp:
                key = int(line.split(" ")[0])
                value = {'Mean_Return': float(line.split(" ")[1]), 'SD': float(line.split(" ")[2].strip('\n'))}
                ReturnSD_dict[key] = value
        return ReturnSD_dict, corr_dict

    def get_cov_dict(self):
        '''Returns a dict of cov[(stocki, stockj):cov_value]
        '''
        cov = dict()
        for pair, corr in self.corr.items():
            cov[pair] = corr * self.ReturnSD[pair[0]]['SD'] * self.ReturnSD[pair[1]]['SD']
        return cov

    def get_initial_solution(self):
        '''
        Creates an initial solution based on a greedy approach.
        Input: mean_return & SD filePath, Correlation filePath, Risk aversion(Lambda), assets in portfolio(k),
        upper and lower bounds (epsilon & delta), number of iterations(T)
        '''
        Initial = CH.GREEDY(ReturnSD_path= self.ReturnSD, corr_path=self.corr ,Lambda= self.Lambda, k= self.k,
                            epsilon= self.epsilon, delta=self.delta, T=1000)
        return Initial.initial_solution, Initial.initial_objvalue

    def get_tabuestructure(self):
        '''Takes a dict (input data)
        Returns a dict of tabu attributes(assets) as keys and [tabu_time] as values
        '''
        tabu_str = dict()
        # Three tabu lists for three moves
        for asset in self.ReturnSD:
            tabu_str[asset] = {"I":0, "D":0, "C":0}
        return tabu_str


    def Objfun(self, solution):
        '''Takes a dict as solution (stock:weight).
        returns the multi objective function value of the solution
        '''
        objvalue_1 = 0   # The first part of the multi_objective fun
        for i in solution:
            for j in solution:
                if (i,j) in self.cov:
                    objvalue_1 += float(solution[i] * solution[j] * self.cov[(i,j)])
                else:
                    objvalue_1 += float(solution[i] * solution[j] * self.cov[(j,i)])
        objvalue_2 = 0   # The second part of the multi_objective fun
        for i in solution:
            objvalue_2 += float(solution[i] * self.ReturnSD[i]['Mean_Return'])

        multi_objvalue = (self.Lambda * objvalue_1) - ((1-self.Lambda) * objvalue_2)
        return multi_objvalue

    def Rescale(self, S):
        '''
        Ensures that all weights in a [S]olution (portfolio) satisfy sum to 1 and lower bound(epsilon) constraints
        '''
        free_prop = 1 - (self.k * self.epsilon)
        S = valmap(lambda x: self.epsilon + ((x / sum(S.values())) * free_prop), S)
        return S

    def I_move(self, solution, a_i, q =0.1):
        '''
        [I]ncrease_move: Takes a dict solution (portfolio)
        returns a new neighbor solution with a given asset's (a_i) weight increased by stepsize q
        '''
        w_i = solution[a_i] * (1+q) # increasing the weight of asset a_i in the solution
        if w_i > self.delta:
            w_i = self.delta # Ensure the upper bound
        
        solution = self.Rescale(solution)
        print(solution)
        return solution





test = TS_ShortMemory(ReturnSD_path= "/home/taylan/PycharmProjects/POP/Data/Hong_Kong_31/Return&SD.txt",
                      corr_path="/home/taylan/PycharmProjects/POP/Data/Hong_Kong_31/correlation.txt",Lambda=0.5,
                      k=10, epsilon=0.00, delta=1)

m = {29: 0.3632909755289824, 5: 0.2627984377653925, 9: 0.19055618129352217, 12: 0.18335440541210293}
n=test.I_move(m,29)