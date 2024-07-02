# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 10:50:59 2024

@author: WWA882
"""

###############################################################################
# Package import
###############################################################################

import gurobipy as gp
from gurobipy import GRB
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import math
import itertools



###############################################################################
# Functions for solving the bin packing problem and computing the Shapley values
###############################################################################


# argument is a specific dataframe iwht information about the bin packing instance, -> see later the specific format
def solve_bin_packing_with_size_and_weight(bin_packing_instance):
    
    # Defining sets of items and bins
    I = [i for i in range(0,len(bin_packing_instance))] # set of items
    B = [i for i in range(0,len(I))] # set of bins -> create as many bins as items because this is the natural upper bound

    # weight and size of items
    w_i = list(bin_packing_instance["item_weight"])
    s_i = list(bin_packing_instance["item_size"])
    
    # bin capacity for weight and size
    bin_weight_capacity = list(bin_packing_instance["bin_weight"])[0]
    bin_size_capacity = list(bin_packing_instance["bin_size"])[0]
    
    
    # Create a new model in gurobipy
    model = gp.Model("Bin_Packing_with_size_and_weight")
    
    # Decision variables: y is bin selection variable and z is item to bin assignment variable
    y = model.addVars(B, vtype=GRB.BINARY, name="y")  # y[b] is 1 if bin b is used
    z = model.addVars(I,B, vtype=GRB.BINARY, name="z")  # z[i, b] is 1 if item i is assigned to bin b
    
    
    #Objective function -> minimize number of selected bins
    model.setObjective(gp.quicksum(y[b] for b in B), GRB.MINIMIZE)
    
    #Constraints

    # Assignment constraints 
    model.addConstrs(gp.quicksum(z[i, b] for b in B) == 1 for i in I) # each item exaclty assigned once
    model.addConstrs(gp.quicksum(z[i, b] for i in I) <= len(I) * y[b] for b in B) # assignment only allowed if bin is active
    
    
    # Size and weight capacity constraints
    model.addConstrs(gp.quicksum(z[i,b] * w_i[i] for i in I ) <= bin_weight_capacity for b in B) # weight 
    model.addConstrs(gp.quicksum(z[i,b] * s_i[i] for i in I ) <= bin_size_capacity for b in B) # size
    
    # Symmetry breaking constraints -> enhances solvers performance a bit, bins are selected in order
    model.addConstrs(y[b] <= y[b-1] for b in range(1, len(B)))
    

    # Solver settings
    model.Params.OutputFlag = 0 # no textoutput -> saves runtime
    model.Params.Presolve = 2  # Aggressive presolve
    model.Params.MIPGap = 0.01 # gaps allowed because of specific problem structure, gap of e.g. 1% already means exactly solved
    
    # Optimize the model
    model.optimize()
    
    # @Dennis: Hab hier noch nen Textoutput falls du an dem Code rumschraubst und Textoutputs haben willst. Habs jetzt mal auskommentiert drin gelassen
    # Print objective value
    #print(f"Objective value (number of bins used): {model.objVal}")
    
    
    # print unsed bins
    #opened_bins = [b for b in B if y[b].X > 0.5]
    #print(f"Opened bins: {opened_bins}")
    
    
    # print assignment of items to bins
    #for b in B:
     #   assigned_items = [i for i in I if z[i, b].X > 0.5]
      #  if assigned_items:
           # print(f"Bin {b} contains items: {assigned_items}")

    
    # Only return objective value, -> could maybe be adapted if features like utilization or similar things are required
    return model.objVal


# Similar Shapley value function as in TSP/CVRP instances, n is the number of items
def shapley_value(player_index, characteristic_function, n):
    
    # initialize with 0
    shapley_value = 0.0
    
    # players start from 1 to n
    players = list(range(1, n+1))
    
    for coalition_size in range(1, n+1 ):
        # create subcoalitions
        for coalition in itertools.combinations(players, coalition_size):
            # classical shapley value procedur
            if player_index in coalition:
                coalition_value = characteristic_function.get(coalition, 0)
                coalition_without_i = set(coalition) - {player_index}
                prev_coalition_value = characteristic_function.get(tuple(sorted(coalition_without_i)), 0)
                marginal_contribution = coalition_value - prev_coalition_value
                
                num_possible_orders = (
                    math.factorial(coalition_size - 1) * math.factorial(n - coalition_size)
                )
                shapley_value += marginal_contribution * (num_possible_orders / math.factorial(n))
    
    # return shapley value
    return shapley_value



###############################################################################
# Create instances
###############################################################################

# parametersetting for instance
instance_id = 0 # later when generating more instances loop over the whole code and adapt instance_id in each iteration

num_items =10 # number of items

# Define bin dimensions
bin_weight_capacity = 10
bin_size_capacity = 10

# Generate random weights and sizes for items
w_i = [random.randint(1, 6) for _ in range(num_items)]
s_i = [random.randint(1, 6) for _ in range(num_items)]

# Create dataframe
bin_packing_instance = pd.DataFrame({
    'item_id': range(1, num_items + 1),
    'item_weight': w_i,
    'item_size': s_i,
    'bin_weight': [bin_weight_capacity] * num_items,
    'bin_size': [bin_size_capacity] * num_items
})

# Reorder columns to have bin_weight and bin_size on the right
bin_packing_instance = bin_packing_instance[['item_id', 'item_weight', 'item_size', 'bin_weight', 'bin_size']]

bin_packing_instance

###############################################################################
# Solve created instance with solve function
###############################################################################
total_bins = solve_bin_packing_with_size_and_weight(bin_packing_instance)

# append total bins to dataframe
bin_packing_instance["Total bins"] = total_bins


###############################################################################
# Determine characteristic function -> necessary for shapley values
###############################################################################

# determine all subsets

list_of_all_subsets = []

for item in range(1, num_items+1):
    subsets = itertools.combinations(range(1, num_items+1), item)
    for subset in subsets:
        list_of_all_subsets.append(subset)

# initialize dictionary with subset total cost / total number of bins
subset_total_cost = {}

# evaluate total cost / total number of bins for each subset and store in dictionary
for subset in list_of_all_subsets:
    
    subset_instance = bin_packing_instance[bin_packing_instance['item_id'].isin(subset)]
    
    if len(subset_instance) == 0:
        total_cost = 0
    else:
        
        total_cost  = solve_bin_packing_with_size_and_weight(subset_instance)
    
    subset_total_cost[subset] = total_cost


###############################################################################
# Determine shapley values based on the characteristic function
###############################################################################

shapley_values = [0]
shapley_sum = 0
for i in range(1,num_items+1):  
    shapley_val = shapley_value(i, subset_total_cost, num_items)        
    shapley_values = shapley_values + [shapley_val]


# append shapley values as columns to dataframe
bin_packing_instance["Shapley Value"] = shapley_values[1:]


bin_packing_instance

###############################################################################
# Generate more features -> Dennis part ;) 
# -> vorallem descriptive statistiken zu size und weight. Wieviele sind größer / kleiner, percentile, summe, gemeinsame Betrachtung von weight & size, eventuell noch utilization of bins. Vielleicht noch irgendwelche Features wie gut das item zu anderen Items passt um einen bin komplett voll zu machen etc... , hauptsache kreativ sein
###############################################################################




###############################################################################
# Save instances as excel & parquet
###############################################################################

bin_packing_instance.to_excel("C:/Users/WWA882/Documents/Wissenschaftlicher_Mitarbeiter/Forschung/Paper_3/Bin_packing_instances/Items_" + str(num_items) + "_id_" + str(instance_id) + ".xlsx")      
bin_packing_instance.to_parquet("C:/Users/WWA882/Documents/Wissenschaftlicher_Mitarbeiter/Forschung/Paper_3/Bin_packing_instances/Items_" + str(num_items) + "_id_" + str(instance_id) + ".parquet")
   






