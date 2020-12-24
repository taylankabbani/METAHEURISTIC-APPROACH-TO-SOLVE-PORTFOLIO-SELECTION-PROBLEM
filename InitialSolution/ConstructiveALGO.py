from toolz import valmap
import numpy as np

class GREEDY():
    '''
    Creates an initial solution based on a greedy approach.
    Input: mean_return & SD filePath, Correlation filePath, Risk aversion(Lambda), assets in portfolio(k),upper and lower bounds (epsilon & delta), number of iterations(T)
    '''
    def __init__ (self, ReturnSD_path, corr_path, Lambda, k,epsilon, delta, T, show=False):
        self.ReturnSD, self.corr = self.get_data(ReturnSD_path, corr_path)
        self.cov = self.get_cov_dict()
        self.Lambda = Lambda
        self.k = k
        self.epsilon = epsilon
        self.delta = delta
        self.T = T
        self.show = show
        self.initial_solution, self.initial_objvalue = self.greedy_fun()
  


    def get_data(self, ReturnSD_path, corr_path):
        '''Returns dict of ReturnSD[stock: {Mean_Return, StandardDeviation}] and a dict of corr[(stocki,stockj:corr_value)]
        '''
        if __name__ == "__main__":
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
        else:
            return ReturnSD_path, corr_path
    
    def get_cov_dict(self):
        '''Returns a dict of cov[(stocki, stockj):cov_value]
        '''
        cov = dict()
        for pair,corr in self.corr.items():
            cov[pair] = corr * self.ReturnSD[pair[0]]['SD'] * self.ReturnSD[pair[1]]['SD']
        return cov

    def get_sharpe_ratio(self):
        '''Sharpe_ration for each stock in an descending order
        '''
        Sharpe_ratio = [[key, value['Mean_Return']/value['SD']] for key, value in self.ReturnSD.items()]
        Sharpe_ratio = sorted(Sharpe_ratio, key = lambda x: x[1], reverse=True)
        return Sharpe_ratio
    
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
        objvalue_2 = 0
        for i in solution:
            objvalue_2 += float(solution[i] * self.ReturnSD[i]['Mean_Return'])

        multi_objvalue = (self.Lambda * objvalue_1) - ((1-self.Lambda) * objvalue_2)
        return multi_objvalue

    def greedy_fun(self):
        Best_solution = {}
        V = float("inf")
        sharpe_ratio = self.get_sharpe_ratio()
        for i in range(self.T):
            # current solution
            S = {}
            for stock in range(self.k):  #Random weights for k top stocks
                S[sharpe_ratio[stock][0]] = np.random.uniform(self.epsilon, self.delta)
            free_prop = 1 - (self.k * self.epsilon)
            S = valmap(lambda x: self.epsilon + ((x/sum(S.values()))*free_prop), S) # Normalizing to sum up to 1
            f = self.Objfun(S) # Objective function value for the current solution
            if f < V : #update best solution (minimizaion)
                if self.show == True:
                    print("### itr:{}  Best_solution: {} , Best_Objvalue: {} \n    Current_solution:{} Current_Objvalue: {} ==> Improving solution found".format( i,Best_solution, V, S, f))
                V = f
                Best_solution = S
            else:
                if self.show == True:
                    print("### itr:{}  Best_solution: {} , Best_Objvalue: {} \n    Current_solution:{} Current_Objvalue: {} ==> No Improving Solution".format( i,Best_solution, V, S, f))
        if self.show == True:
            print("#"*50, "Best Solution found: {}\nObjvalue: {}\nNumber of iter: {}".format(Best_solution, V, self.T), "#"*50, sep="\n")
        return Best_solution, V

# test = GREEDY(ReturnSD_path= "/home/taylan/PycharmProjects/POP/Data/Hong_Kong_31/Return&SD.txt",
#               corr_path="/home/taylan/PycharmProjects/POP/Data/Hong_Kong_31/correlation.txt",
#               Lambda=0.5, k= 10 , epsilon=0.01, delta=1, T=1000, show=True)



