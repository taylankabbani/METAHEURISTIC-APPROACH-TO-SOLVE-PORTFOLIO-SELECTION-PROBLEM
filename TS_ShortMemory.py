from toolz import valmap, valfilter
import numpy as np
from InitialSolution import ConstructiveALGO as CH
import random as rd

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
        Returns a dict of tabu attributes(asset) as keys and {tabu_time,move_value} as values
        '''
        tabu_str = dict()
        # Three tabu lists for three moves
        inf = float('inf')
        for asset in self.ReturnSD:
            tabu_str[asset] = {'tabu_time':{"I":0, "D":0, "S":0}, 'move_val':{"I":inf, "D":inf, "S":inf}}
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

    def Rescale(self, solution):
        '''
        Ensures that all weights in a [S]olution (portfolio) satisfy the sum to 1, lower(epsilon) and upper(delta) bound
        constraints
        '''
        free_prop = 1 - (self.k * self.epsilon)
        solution = valmap(lambda x: self.epsilon + ((x / sum(solution.values())) * free_prop), solution)
        # Fixing upper bond:
        R = valfilter(lambda x: x > self.delta, solution) # assets with exceeded weights
        if len(R) > 0:
            L = sum(valfilter(lambda x: x < self.delta, solution).values()) # sum of weights less than delta
            free_prop = 1 - (self.k * self.epsilon + len(R)*self.delta) # Free proportion in the solution
            solution = valmap(lambda x: self.epsilon + (x*free_prop/L) if x < self.delta else self.delta, solution)
        return solution

    def I_move(self, solution, i, q):
        '''
        [I]ncrease move: Takes a dict solution (portfolio)
        returns a new neighbor solution with a given asset's (i) weight increased by stepsize q
        '''
        solution = solution.copy()
        w_i = solution[i] * (1+q) # increasing the weight of asset i in the solution
        solution[i] = w_i
        
        solution = self.Rescale(solution)
        return solution

    def D_move(self, solution, i, q):
        '''
        [D]ecrease move: Takes a dict solution (portfolio)
        returns a new neighbor solution with a given asset's (i) weight decreased by stepsize q
        If the asset's weight falls bellow epsilon it is replaced with anther asset not yet in solution
        '''
        solution=solution.copy()
        w_i = solution[i] * (1-q) # decrease the weight of asset i in the solution
        if w_i < self.epsilon:
            while True:
                j = rd.choice(list(self.ReturnSD.keys())) # randomly selecting asset j not in solution
                if j in solution:
                    continue
                else:
                    break
            solution.pop(i)
            solution[j] = self.epsilon # add the randomly chosen asset to solution and set weight to epsilon
        else:
            solution[i] = w_i
        solution = self.Rescale(solution)
        return solution

    def S_move(self, solution, j):
        '''
        [S]wap move: Takes a dict solution (portfolio)
        returns a new neighbor solution with an asset (j) not in solution swapped with i asset in the solution that
        has the minimum weight.
        '''
        solution = solution.copy()
        if j not in solution:
            i = min(solution,key=lambda x: solution[x])
            w_i = solution[i] # asset i's weight
            solution.pop(i)
            solution[j] =w_i
            return solution
        else:
            return "Error: J in solution"

    def T1_runner(self,solution,q):
        '''
        Token ring search: Takes a dict solution (portfolio), stepsize q.
        Searches the whole neighborhood of the three move function (I,D and S) using t1 runner and updates move
        values accordingly.
        '''
        # Resetting all move values to inf
        for asset in self.tabu_str:
            for move_fun in self.tabu_str[asset]['move_val']:
                self.tabu_str[asset]['move_val'][move_fun] = float('inf')
        for asset in solution:
            self.tabu_str[asset]['move_val']['I'] = self.Objfun(self.I_move(solution,asset,q))# Neighborhood solutions by [I]ncrease Move
            self.tabu_str[asset]['move_val']['D'] = self.Objfun(self.D_move(solution,asset,q))# Neighborhood solutions by [D]ncrease Move
        for asset in set(self.ReturnSD.keys()) - set(solution):
            self.tabu_str[asset]['move_val']['S'] = self.Objfun(self.S_move(solution,asset))# Neighborhood solutions by [S]wap Move


    def TSearch(self):
        '''
        The implementation Tabu search algorithm with short-term memory.
        '''
        # Parameters:
        tenure = 3
        best_solution = self.initial_solution
        best_objvalue = self.initial_objvalue
        current_solution = self.initial_solution
        current_objvalue = self.initial_objvalue
        print("#" * 30, "Short-term memory TS", "#" * 30,
              "\nInitial Solution: {}, Initial Objvalue: {}".format(current_solution, current_objvalue), sep='\n\n')
        iter = 1
        Terminate = 0
        while iter < 1000:
            q = 0.1
            print('\n\n### iter {}###  Current_Objvalue: {}, Best_Objvalue: {}'.format(iter, current_objvalue,best_objvalue))

            self.T1_runner(current_solution,q) # Searching whole neighborhoods using T1 runner
            # Admissible move
            while True:
                # Select the best move from the neighborhood union of three move functions
                t1_bestmoves = {'I':[0,0],'D':[0,0],'S':[0,0]} # move_fun:[move_val, asset]
                for move_fun in t1_bestmoves:
                    # Best move values obtained by each move function
                    t1_bestmoves[move_fun][1] = min(self.tabu_str, key=lambda x: self.tabu_str[x]['move_val'][
                        move_fun])
                    t1_bestmoves[move_fun][0]= self.tabu_str[t1_bestmoves[move_fun][1]]['move_val'][move_fun]

                t1_fun = min(t1_bestmoves,key=lambda x: t1_bestmoves[x][0]) # Best move function
                t1_a =  t1_bestmoves[t1_fun][1] # The asset moved
                t1_val = t1_bestmoves[t1_fun][0] # Best move value
                t1_tabutime = self.tabu_str[t1_a]['tabu_time'][t1_fun] # Tabu Time

                # Not Tabu
                if t1_tabutime < iter:
                    # make the move
                    current_solution = self.I_move(current_solution, t1_a, q) if t1_fun == "I" else self.D_move(
                        current_solution, t1_a, q) if t1_fun == "D" else self.S_move(current_solution, t1_a)
                    current_objvalue = t1_val
                    # Best Improving move
                    if  t1_val < best_objvalue:
                        best_solution = current_solution
                        best_objvalue = current_objvalue
                        print("   Candidate Move: {} asset {}, Objvalue: {} => Best Improving => Admissible".format(
                            t1_fun, t1_a,current_objvalue), current_solution)
                        Terminate = 0
                    else:
                        print("   Candidate Move: {} asset {}, Objvalue: {} => Least non-improving => "
                              "Admissible".format(t1_fun, t1_a,current_objvalue), current_solution)
                        Terminate += 1
                    # update tabu_time for the move
                    self.tabu_str[t1_a]['tabu_time'][t1_fun] = iter + tenure
                    iter += 1
                    break

                # If tabu
                else:
                    # Aspiration
                    if t1_val < best_objvalue:
                        # make the move
                        best_solution = current_solution = self.I_move(current_solution, t1_a, q) if t1_fun == "I" else\
                            self.D_move(current_solution, t1_a, q) if t1_fun == "D" else self.S_move(current_solution, t1_a)
                        best_objvalue = current_objvalue = t1_val
                        print("   Candidate Move: {} asset {}, Objvalue: {} => Aspiration => Admissible".
                              format(t1_fun,t1_a,current_objvalue), current_solution)
                        Terminate = 0
                        iter += 1
                        break
                    # If the move is tabu & aspiration is not met
                    else:
                        self.tabu_str[t1_a]['move_val'][t1_fun] = float('inf')
                        print("   Candidate Move: {} asset {}, Objvalue: {} => Tabu => Inadmissible".
                              format(t1_fun,t1_a,t1_val))
                        continue







test = TS_ShortMemory(ReturnSD_path= "/home/taylan/PycharmProjects/POP/Data/Hong_Kong_31/Return&SD.txt",
                      corr_path="/home/taylan/PycharmProjects/POP/Data/Hong_Kong_31/correlation.txt",Lambda=0.6,
                      k=4, epsilon=0.01, delta=1)

test.TSearch()


# initial_val = test.initial_objvalue

# m = test.I_move(initial,29)
# m_val = test.Objfun(m)
# m = test.D_move(initial,12)


