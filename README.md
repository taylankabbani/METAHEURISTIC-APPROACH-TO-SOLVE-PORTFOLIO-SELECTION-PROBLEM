# METAHEURISTIC APPROACH TO SOLVE PORTFOLIO SELECTION PROBLEM
### [Research Paper](https://github.com/taylankabbani/METAHEURISTIC-APPROACH-FOR-POP/blob/master/Description/Progress_Repo.pdf)
 Tabu Search and TokenRing Search is being used in order to solve the Portfolio Optimization Problem. The seminal mean-variance model of Markowitz is being considered with the addition of cardinality and quantity constraints to better capture the dynamics of the trading procedure, the model becomes NP-hard problem which can not be solved using an exact method. The combination of three different neighborhood relations is being explored with Tabu Search. In addition, a new constructive method for the initial solution is proposed. Finally, I show how the proposed techniques perform on public benchmarks.
 
## [Proposed Constructive Heuristics](https://github.com/taylankabbani/METAHEURISTIC-APPROACH-TO-SOLVE-PORTFOLIO-SELECTION-PROBLEM/blob/master/TS_TokenRing.py): 
![](https://github.com/taylankabbani/METAHEURISTIC-APPROACH-TO-SOLVE-PORTFOLIO-SELECTION-PROBLEM/blob/master/Description/Algo1.PNG)


## [TabuSearch + TokenRing Search](https://github.com/taylankabbani/METAHEURISTIC-APPROACH-TO-SOLVE-PORTFOLIO-SELECTION-PROBLEM/tree/master/InitialSolution): 
### T1 Runner
![](https://github.com/taylankabbani/METAHEURISTIC-APPROACH-TO-SOLVE-PORTFOLIO-SELECTION-PROBLEM/blob/master/Description/Algo2.PNG)

### T2 Runner
![](https://github.com/taylankabbani/METAHEURISTIC-APPROACH-TO-SOLVE-PORTFOLIO-SELECTION-PROBLEM/blob/master/Description/Algo3.PNG)

# Results
|   Index   	| Assets 	|                                                                	|         TS&TokenRing         	|       TS In Ref      	|
|:---------:	|:------:	|:--------------------------------------------------------------:	|:----------------------------:	|:-----------------------:	|
| Hang Seng 	|   31   	|  Median percentage error<br>Mean percentage error<br>Time (s)  	|   1.812<br>2.2656<br>1154.3  	|  1.2181<br>1.1217<br>74 	|
|    DAX    	|   85   	| Median percentage error <br>Mean percentage error <br>Time (s) 	|     4.21<br>4.035<br>2873    	| 2.6380<br>3.3049<br>199 	|
|    FTSE   	|   89   	| Median percentage error <br>Mean percentage error <br>Time (s) 	|   1.2406<br>1.2959<br>2919   	| 1.0841<br>1.6080<br>246 	|
|    S&P    	|   98   	| Median percentage error <br>Mean percentage error <br>Time (s) 	|   2.3630<br>2.5068<br>3107   	| 1.2882<br>3.3092<br>225 	|
|   Nikkei  	|   225  	| Median percentage error <br>Mean percentage error <br>Time (s) 	| 1.34635<br>1.21220<br>5866.2 	| 0.6093<br>0.8975<br>545 	|

