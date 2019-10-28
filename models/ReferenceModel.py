
from pyomo.environ import *
import numpy as np

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

model.p_da_act = Param(model.T, within=PositiveReals)

model.gradient = Param(within=PositiveReals)



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


#
# Stage-specific cost computations
#

def ComputeFirstStageCost_rule(model):
    return 0

model.FirstStageCost = Expression(rule=ComputeFirstStageCost_rule)

def ComputeSecondStageCost_rule(model):
    return sum(model.u_sch[t] * model.p_da_act[t] for t in model.T)

model.SecondStageCost = Expression(rule=ComputeSecondStageCost_rule)


#
# Objectives
#

def day_ahead_obj_rule(model):
    return sum(model.u_sch[t] * model.p_da_pred[t] for t in model.T)

model.day_ahead_rule = Objective(rule=day_ahead_obj_rule)

