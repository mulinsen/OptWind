# -*- coding: utf-8 -*-
from WindFarm.wind_turbine import WindTurbine
from WindFarm.wind_farm_design import WindFarmDesign
from WindFarm.site_condition import SiteCondition
from WindFarm.flow_field import FlowField
from WindFarm.AEP import AEP
from WindFarm.optimization import Optimization
from WindFarm.constraints import Constraints
import pandas
import numpy as np

##############################################################################
# 1. Initialization 
grd_path = '../inputs/Middelgrunden/'
site_condition = SiteCondition.from_WAsP_grd(grd_path)

x_min = site_condition.x_range[0]
x_max = site_condition.x_range[-1]
y_min = site_condition.y_range[0]
y_max = site_condition.y_range[-1]

wtg_file = '../inputs/Middelgrunden/Bonus 2 MW.wtg'
wind_turbine = WindTurbine.from_WAsP_wtg(wtg_file)

layout_file = '../inputs/Middelgrunden/Sites from MG.txt'
df = pandas.read_table(layout_file, header=None)
num_wt = df.shape[0]
complete_layout = np.zeros((num_wt*1, 4))
complete_layout[:, :3] = df.values[:, 1:]

# choose different layout as starting point
initial_layout = 'original' # choose in ['original', 'grid_fill', 'random']

if initial_layout == 'grid_fill':
    x_array, y_array = np.meshgrid(
            np.linspace(x_min, x_max, 4), np.linspace(y_min, y_max, 5))
    complete_layout[:, 0] = x_array.flatten()
    complete_layout[:, 1] = y_array.flatten()
elif initial_layout == 'random':
    complete_layout[:, 0] = x_min + np.random.random(num_wt) * (x_max - x_min)
    complete_layout[:, 1] = y_min + np.random.random(num_wt) * (y_max - y_min)
    
wind_farm_design = WindFarmDesign(complete_layout, [wind_turbine],
                                  name='Middelgrunden')

flow_field = FlowField(wind_farm_design, site_condition.getAkf,
                       site_condition.height_ref,
                       ws_binned=np.linspace(1, 30, 30),
                       wd_binned=np.linspace(0, 354, 60)          
                       )
flow_field.cal_flow_field()

wind_farm_constraint = Constraints(x_min, x_max, y_min, y_max)

opt_problem = Optimization(wind_farm_design, flow_field, wind_farm_constraint,
                           site_condition, AEP, 
                           save_path = '../outputs/Middelgrunden')


opt_problem.random_search_run(maximal_evaluations=1000, maximal_step_size=100)
opt_problem.random_search_run(maximal_evaluations=1000, maximal_step_size=1000)
opt_problem.random_search_run(maximal_evaluations=1000, maximal_step_size=2000)  
opt_problem.random_search_run(maximal_evaluations=1000, maximal_step_size=4000)
