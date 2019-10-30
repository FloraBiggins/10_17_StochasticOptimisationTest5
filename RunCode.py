import os

myCmd = 'runef -m models -s nodedata --solve --solver=gurobi --solution-writer=pyomo.pysp.plugins.csvsolutionwriter'

os.system(myCmd)

import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_csv('ef.csv')

prices = results.iloc[145:647,4]

cvar = results.iloc[120,4]

print(cvar)

num_bins = 20
n, bins, patches = plt.hist(prices, num_bins, facecolor='blue', alpha=0.5)
plt.axvline(cvar, color='k', linestyle='dashed', linewidth=1)
plt.text(cvar - 900, 50, 'CVaR: {:.2f}'.format(cvar))
plt.show()