import os

myCmd = 'runef -m models -s nodedata --solve --solver=gurobi --solution-writer=pyomo.pysp.plugins.csvsolutionwriter'

os.system(myCmd)

import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_csv('ef.csv')

prices = results.iloc[145:647,4]

var = results.iloc[119,4]
cvar = results.iloc[120,4]
pred_cost = results.iloc[145,4]

print(cvar)

num_bins = 20
n, bins, patches = plt.hist(prices, num_bins, facecolor='blue', alpha=0.4)
plt.axvline(cvar, color='k', linestyle='dashed', linewidth=1)
plt.text(cvar-20, 55, 'CVaR', fontsize=10, bbox = {'facecolor' : 'white'}) #: {:.2f}.format(cvar)
plt.axvline(var, color='k', linestyle='dashed', linewidth=1)
plt.text(var-180, 55, 'VaR', fontsize=10, bbox = {'facecolor' : 'white'})
plt.axvline(pred_cost, color='r', linestyle='dashed', linewidth=1)
plt.text(pred_cost-180, 5, 'Predicted \n Cost', fontsize=10, color='red')
plt.xlabel('Day-Ahead Market Generated Cost Scenarios (£)')
plt.ylabel('Frequency')
plt.show()