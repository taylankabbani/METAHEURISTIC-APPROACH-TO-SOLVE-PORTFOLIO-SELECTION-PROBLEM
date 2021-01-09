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

solutions, portcef = Solutions("Data/Japan_225/Return&SD.txt", "Data/Japan_225/correlation.txt")
# Unconstrained Efficient frontier
portef = pd.read_csv("Data/Japan_225/portef.txt", sep='\t', header=None,names = ['Return', 'Risk'])


# Variance error
ef_variances = []

for i in portcef.Return:
    col = portef[round(portef.Return,6) == round(i,6)].values.tolist()
    if len(col)== 0:
        col = portef[round(portef.Return,5) == round(i,5)].values.tolist()
        if len(col)== 0:
                col = portef[round(portef.Return, 4) == round(i, 4)].values.tolist()
                if len(col) == 0:
                    col = portef[round(portef.Return, 3) == round(i, 3)].values.tolist()
    sum = 0
    for item in col:
        sum += item[1]
    ef_Risk = sum/len(col)

    ef_variances.append(ef_Risk)
# Return Error
ef_returns = []
for i in portcef.Risk:
    col = portef[round(portef.Risk,6) == round(i,6)].values.tolist()
    if len(col)== 0:
        col = portef[round(portef.Risk,5) == round(i,5)].values.tolist()
        if len(col)== 0:
                col = portef[round(portef.Return, 4) == round(i, 4)].values.tolist()
                if len(col) == 0:
                    col = portef[round(portef.Return, 3) == round(i, 3)].values.tolist()
    sum = 0
    for item in col:
        sum += item[0]
    ef_Return = sum/len(col)
    ef_returns.append(ef_Return)
    
portcef['ef_Return'] =  ef_returns
portcef['ef_Risk'] = ef_variances

# Return Error
portcef = portcef.assign(Return_Error = lambda portcef : 100 * (portcef.ef_Return - portcef.Return) / portcef.ef_Return)
#Variance Error
portcef = portcef.assign(Risk_Error = lambda portcef : 100 * (portcef.Risk - portcef.ef_Risk) /portcef.ef_Risk)

print("Variance of return error: {}\nMean return error: {}".format(round(portcef.Risk_Error.mean(),4),
round(portcef.Return_Error.mean(),4)))

portcef.plot(x = "Risk", y ='Return')
plt.xlim([0, 0.006]);
plt.ylim([0, 0.013]);
plt.savefig('portcef')

portef.plot(x = "Risk", y ='Return')
plt.xlim([0, 0.006]);
plt.ylim([0, 0.013]);
plt.savefig('portef')