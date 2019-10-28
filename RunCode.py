import os

myCmd = 'runef -m models -s nodedata --solve --solver=gurobi --generate-weighted-cvar --risk-alpha=0.8 --cvar-weight=5'

os.system(myCmd)