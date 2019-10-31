
from pyomo.environ import *
import random

#
# Model
#

model = AbstractModel()

#
# Parameters
#

model.T= Set()

model.i = Set()

model.capacity = Param(model.i, within=PositiveReals)

model.power = Param(model.i, within=PositiveReals)

model.l_SOC = Param(model.i, within=PositiveReals)

model.u_SOC = Param(model.i, within=PositiveReals)

model.efficiency_c = Param(model.i, within=PositiveReals)

model.efficiency_d = Param(model.i, within=PositiveReals)

model.l_pred = Param(model.T, within=PositiveReals)

model.p_da_pred = Param(model.T, within=PositiveReals)

model.gradient = Param(within=PositiveReals)

model.alpha = Param(initialize=0.95, within=PositiveReals)

model.beta = Param(initialize=10, within=PositiveReals)

price_scenarios = 200

load_scenarios = 10

#
# Variables
#

# Energy Storage Variable

def capacity_bounds_rule(model, t, i):
    return (model.l_SOC[i] * model.capacity[i], model.u_SOC[i] * model.capacity[i])

model.x = Var(model.T, model.i, bounds=capacity_bounds_rule)

def power_bounds_rule(model, t, i):
    return (0, model.power[i])

model.c = Var(model.T, model.i, bounds=power_bounds_rule)

model.d = Var(model.T, model.i, bounds=power_bounds_rule)

model.u_sch = Var(model.T)

# Price Variables

model.p_pm_pred = Var(model.T)

model.p_da_act = Var(model.T)

model.prices = Var(range(price_scenarios), range(load_scenarios))

model.pred_cost = Var()

# Load Variables

model.load_scenarios = Var(range(load_scenarios), model.T)

model.l_act = Var(model.T)

# CVaR Variables

model.cvar = Var()

model.z = Var()

model.y = Var(range(price_scenarios), range(load_scenarios))


#
# Constraints
#


# Storage Constraints


def initial_soc_rule(model, i):
    return model.x[23,i] == model.l_SOC[i] * model.capacity[i]

model.initial_soc = Constraint(model.i, rule=initial_soc_rule)

def soc_rule(model, t, i):
    if t == 0:
        s = 23
    else:
        s = t - 1
    return model.x[t,i] == model.x[s,i] + model.efficiency_c[i] * model.c[t,i] - model.d[t,i]

model.soc_constraint = Constraint(model.T, model.i, rule=soc_rule)

def u_scheduling_rule(model, t):
    return model.u_sch[t] == (model.l_pred[t]/1000) + sum(model.c[t,i] - model.efficiency_d[i] *
                                                          model.d[t,i] for i in model.i)

model.u_scheduling_constraint = Constraint(model.T, rule=u_scheduling_rule)

def p_pm_rule(model, t):
    return model.p_pm_pred[t] == model.p_da_pred[t] + model.gradient * model.u_sch[t]

model.p_pm_constraint = Constraint(model.T, rule=p_pm_rule)


# Price Prediction, Scenario Matrix and Actual Price Scenario

def cost_prediction_rule(model):
    return model.pred_cost == sum(model.p_da_pred[t] * model.u_sch[t] for t in model.T)

model.cost_prediction = Constraint(rule=cost_prediction_rule)

def random_price_matrix_rule(model, s, t):
    return model.p_da_pred[t] * random.uniform(0.95,1.05) + random.uniform(-0.05,0.05)

model.price_scenarios = Param(range(price_scenarios), model.T, initialize=random_price_matrix_rule)

def random_price_rule(model, t):
    return model.p_da_act[t] == model.p_da_pred[t] * random.uniform(0.95,1.05) + random.uniform(-0.05,0.05)

model.p_da_act_constraint = Constraint(model.T, rule=random_price_rule)


# Load Scenario Matrix and Actual Load

def random_load_matrix_rule(model, s, t):
    return model.load_scenarios[s, t] == (model.l_pred[t] * random.uniform(0.95,1.05) + random.uniform(-0.05,0.05))/1000 + \
           sum(model.c[t, i] - model.efficiency_d[i] * model.d[t, i] for i in model.i)

model.load_scenarios_constraint = Constraint(range(load_scenarios), model.T, rule=random_load_matrix_rule)


def random_load_rule(model, t):
    return model.l_act[t] == model.l_pred[t] * random.uniform(0.95,1.05) + random.uniform(-0.05,0.05)

model.l_act_constraint = Constraint(model.T, rule=random_load_rule)


# Total Cost Scenario Matrix for All Possible Loads and Prices

def prices_calculation_rule(model, s1, s2):
    return model.prices[s1,s2] == sum(model.price_scenarios[s1,t] * model.load_scenarios[s2,t] for t in model.T)

model.price_calculations = Constraint(range(price_scenarios), range(load_scenarios), rule=prices_calculation_rule)


# CVaR Constraints and Calcuation

def cvar_first_rule(model, i, j):
    return model.prices[i,j] - model.z - model.y[i,j] <= 0

model.cvar_first_constraint = Constraint(range(price_scenarios), range(load_scenarios), rule=cvar_first_rule)

def cvar_second_rule(model, i, j):
    return model.y[i, j] >= 0

model.cvar_second_constraint = Constraint(range(price_scenarios), range(load_scenarios), rule=cvar_second_rule)

def cvar_calculation_rule(model):
    return model.cvar == (model.z +
                          (1/(1 - model.alpha)) * 1/(price_scenarios + load_scenarios) *
                          (sum(model.y[i,j] for i in range(price_scenarios) for j in range(load_scenarios))))

model.cvar_calculation_constraint = Constraint(rule=cvar_calculation_rule)


#
# Stage-specific Cost Computations
#

def ComputeFirstStageCost_rule(model):
    return 0

model.FirstStageCost = Expression(rule=ComputeFirstStageCost_rule)

def ComputeSecondStageCost_rule(model):
    return sum(((model.l_act[t]/1000) + \
               sum(model.c[t,i] - model.efficiency_d[i] * model.d[t,i] for i in model.i)) * \
               model.p_da_act[t] for t in model.T)

model.SecondStageCost = Expression(rule=ComputeSecondStageCost_rule)


#
# Objectives
#

def day_ahead_obj_rule(model):
    return sum(model.u_sch[t] * model.p_da_pred[t] for t in model.T) + \
           model.beta * (model.z + \
                         (1/(1 - model.alpha)) * 1/(price_scenarios + load_scenarios) *
                         sum(model.y[i,j] for i in range(price_scenarios) for j in range(load_scenarios)))

model.day_ahead_rule = Objective(rule=day_ahead_obj_rule, sense=minimize)


