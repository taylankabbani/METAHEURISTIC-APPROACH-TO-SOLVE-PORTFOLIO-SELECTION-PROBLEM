import TS_TokenRing as TS
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time as t


def Solutions(ReturnSD_path, corr_path):
    '''
    Finds the Best Solution (Portfolio) for each value of lambda (51 values)
    '''
    Lambda_values =  np.arange(0 ,1.02, 0.02) # lambda values to be evaluated
    Solutions = []
    Return = []
    Risk = []
    for Lambda_value in Lambda_values:
        T_POP = TS.POP(ReturnSD_path= ReturnSD_path ,
                            corr_path= corr_path ,Lambda= round(Lambda_value,3),
                            k=10, epsilon=0.01, delta=1)
        Best_solution, Best_objval = T_POP.t2_runner()
        obj1 = T_POP.Objfun(Best_solution, Return='Obj1') # Calculating Risk involved with portfolio
        obj2 = T_POP.Objfun(Best_solution, Return='Obj2') # Calculating Return expected of portfolio
        Solutions.append(Best_solution)
        Risk.append(round(obj1,10))
        Return.append(round(obj2,10))
    df = pd.DataFrame(list(zip(Return, Risk)), columns=['Return', 'Risk'])
    # Save results
    df.to_csv('Data/portcef.csv', index=None)
    return Solutions, df


def Error_measures(portcef,portef):
    '''
    linear interpolation in the standard efficient frontier
    '''
    # corresponding Variance for fixed Return
    ef_variances = []
    for i in portcef.Return:
        col = portef[round(portef.Return,6) == round(i,6)].values.tolist()
        if len(col)== 0:
            col = portef[round(portef.Return,5) == round(i,5)].values.tolist()
            if len(col)== 0:
                col = portef[round(portef.Return, 4) == round(i, 4)].values.tolist()
                if len(col) == 0:
                    col = portef[round(portef.Return, 3) == round(i, 3)].values.tolist()
        if len(col) > 1:
            # Risk = y_k + (y_j -y_k) * ((x_i -x_k)/(x_j - x_k))
            Risk = col[0][1] + (col[-1][1] - col[0][1]) * ((i - col[0][0]) / (col[-1][0] - col[0][0]))
        else:
            Risk= col[0][1]
        ef_variances.append(Risk)

    # corresponding Return for fixed Variance
    ef_returns = []
    for i in portcef.Risk:
        col = portef[round(portef.Risk,6) == round(i,6)].values.tolist()
        if len(col)== 0:
            col = portef[round(portef.Risk,5) == round(i,5)].values.tolist()
            if len(col) == 0:
                col = portef[round(portef.Return, 4) == round(i, 4)].values.tolist()
                if len(col) == 0:
                    col = portef[round(portef.Return, 3) == round(i, 3)].values.tolist()
        if len(col) > 1:
            # Return = x_k + (x_j -x_k) * ((y_i -y_k)/(y_j - y_k))
            Return = col[0][0] +(col[-1][0] - col[0][0])*((i-col[0][1])/(col[-1][1]-col[0][1]))
        else:
            Return = col[0][0]
        ef_returns.append(Return)

    portcef['ef_Return'] =  ef_returns
    portcef['ef_Risk'] = ef_variances

    # Return Error
    portcef = portcef.assign(Return_Error = lambda portcef : abs(100 * (portcef.Return- portcef.ef_Return) /
    portcef.ef_Return))
    #Variance Error
    portcef = portcef.assign(Risk_Error = lambda portcef : 100 * (portcef.Risk - portcef.ef_Risk) /portcef.ef_Risk)
    Risk_Error = round(portcef.Risk_Error.mean(),4)

    # The minimum of x-direction, y-direction percentage deviation
    portcef['min'] = portcef[['Return_Error', 'Risk_Error']].min(axis=1)

    Median = portcef['min'].median()
    Mean = portcef['min'].mean()

    print("Median Percentage Error: {}\nMean Percentage Error: {}".format(Median,Mean))

    return Median, Mean

########## Hong_Kong 31 asset#############

# Portfolio constrained efficient frontier
# start = t.time()
# solutions, portcef = Solutions("Data/Hong_Kong_31/Return&SD.txt", "Data/Hong_Kong_31/correlation.txt")
# end = t.time()
# portcef.plot(x = "Risk", y ='Return')
# plt.xlim([0, 0.006]);
# plt.ylim([0, 0.013]);

# Portfolio unconstrained efficient frontier
# portef = pd.read_csv("Data/Hong_Kong_31/portef.txt", sep='\t', header=None,names = ['Return', 'Risk'])
# portef.plot(x = "Risk", y ='Return')
# plt.xlim([0, 0.006]);
# plt.ylim([0, 0.013]);

# MedianError_31, MeanError_31 = Error_measures(portcef,portef)

########## Germany 85 asset#############
# Portfolio constrained efficient frontier
# start = t.time()
# solutions, portcef = Solutions("Data/Germany_85/Return&SD.txt", "Data/Germany_85/correlation.txt")
# end = t.time()
# print('Times:{}'.format(end-start))
# # Portfolio unconstrained efficient frontier
# portef = pd.read_csv("Data/Germany_85/portef.txt", sep='\t', header=None,names = ['Return', 'Risk'])
#
# MedianError_85, MeanError_85 = Error_measures(portcef,portef)

########## Germany 85 asset#############

# Portfolio constrained efficient frontier
start = t.time()
solutions, portcef = Solutions("Data/Japan_225/Return&SD.txt", "Data/Japan_225/correlation.txt")
end = t.time()
print('Times:{}'.format(end-start))
# Portfolio unconstrained efficient frontier
portef = pd.read_csv("Data/Japan_225/portef.txt", sep='\t', header=None,names = ['Return', 'Risk'])

MedianError_225, MeanError_225 = Error_measures(portcef,portef)