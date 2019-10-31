
from pyomo.environ import *
import numpy as np
import random
import matplotlib.pyplot as plt

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

scenarios = 500


#
# Variables
#

def capacity_bounds_rule(model, t, i):
    return (model.l_SOC[i] * model.capacity[i], model.u_SOC[i] * model.capacity[i])

model.x = Var(model.T, model.i, bounds=capacity_bounds_rule)

def power_bounds_rule(model, t, i):
    return (0, model.power[i])

model.c = Var(model.T, model.i, bounds=power_bounds_rule)

model.d = Var(model.T, model.i, bounds=power_bounds_rule)

model.u_sch = Var(model.T)

model.p_pm_pred = Var(model.T)

model.p_da_act = Var(model.T)

model.z = Var()

model.prices = Var(range(scenarios))

model.cvar = Var()

model.y = Var(range(scenarios))

model.pred_cost = Var()


#
# Constraints
#

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
    return model.p_pm_pred[t] == model.p_da_pred[t] + model.gradient * ((model.l_pred[t]/1000) +
                                                            sum(model.c[t,i] - model.efficiency_d[i]
                                                                * model.d[t,i] for i in model.i))

model.p_pm_constraint = Constraint(model.T, rule=p_pm_rule)

def cost_prediction_rule(model):
    return model.pred_cost == sum(model.p_da_pred[t] * model.u_sch[t] for t in model.T)

model.cost_prediction = Constraint(rule=cost_prediction_rule)


def random_price_matrix_rule(model, s, t):
    return model.p_da_pred[t] * random.uniform(0.95,1.05) + random.uniform(-0.05,0.05)

model.scenarios = Param(range(scenarios), model.T, initialize=random_price_matrix_rule)


def cvar_first_rule(model, i):
    return sum(model.scenarios[i,t] * model.u_sch[t] for t in model.T) - model.z - model.y[i] <= 0

model.cvar_first_constraint = Constraint(range(scenarios), rule=cvar_first_rule)

def cvar_second_rule(model, i):
    return model.y[i] >= 0

model.cvar_second_constraint = Constraint(range(scenarios), rule=cvar_second_rule)


def random_price_rule(model, t):
    return model.p_da_act[t] == model.p_da_pred[t] * random.uniform(0.95,1.05) + random.uniform(-0.05,0.05)

model.p_da_act_constraint = Constraint(model.T, rule=random_price_rule)


def prices_calculation_rule(model, i):
    return model.prices[i] == sum(model.scenarios[i,t] * model.u_sch[t] for t in model.T)

model.prices_constraint = Constraint(range(scenarios), rule=prices_calculation_rule)

def cvar_calculation_rule(model):
    return model.cvar == (model.z +
                          (1/(1 - model.alpha)) * (1/scenarios) * (sum(model.y[i] for i in range(scenarios))))

model.cvar_calculation_constraint = Constraint(rule=cvar_calculation_rule)

#
# Stage-specific cost computations
#

def ComputeFirstStageCost_rule(model):
    return model.z + (1/(1 - model.alpha)) * 1/scenarios * (sum(model.y[i] for i in range(scenarios)))

model.FirstStageCost = Expression(rule=ComputeFirstStageCost_rule)

def ComputeSecondStageCost_rule(model):
    return sum(model.u_sch[t] * model.p_da_act[t] for t in model.T)

model.SecondStageCost = Expression(rule=ComputeSecondStageCost_rule)


#
# Objectives
#

def day_ahead_obj_rule(model):
    return sum(model.u_sch[t] * model.p_da_pred[t] for t in model.T) + \
           model.beta * (model.z + \
                         (1/(1 - model.alpha)) * 1/scenarios * sum(model.y[i] for i in range(scenarios)))

model.day_ahead_rule = Objective(rule=day_ahead_obj_rule, sense=minimize)


