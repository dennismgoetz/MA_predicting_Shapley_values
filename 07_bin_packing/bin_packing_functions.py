###############################################################################
# Import packages
###############################################################################
import os
import sys
import random
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import itertools

# Load funtions from file data_gen_functions
current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)

# Join the parent directory with the subfolder that contains the python file with the functions
new_directory = os.path.join(parent_directory, '01_data_generation')

# Add the new directory to the Python path
sys.path.append(new_directory)

from data_gen_functions import fun_shapley_value, fun_convert_time, fun_save_file



###############################################################################
# Functions for solving the bin packing problem and computing the Shapley values
###############################################################################
# Function to generate a bin packing instance
def generate_bin_packing_instance(num_items):
    # Define bin dimensions
    bin_weight_capacity = 10
    bin_size_capacity = 10

    # Generate random weights and sizes for items
    w_i = [random.randint(1, 6) for _ in range(num_items)]
    s_i = [random.randint(1, 6) for _ in range(num_items)]

    # Create dataframe
    bin_packing_instance = pd.DataFrame({
        'Item ID': range(1, num_items + 1),
        'Item Weight': w_i,
        'Item Size': s_i,
        'Bin Weight': [bin_weight_capacity] * num_items,
        'Bin Size': [bin_size_capacity] * num_items
    })

    return bin_packing_instance

# Argument is a specific dataframe with information about the bin packing instance -> see later the specific format
def solve_bin_packing_with_size_and_weight(bin_packing_instance):
    
    # Defining sets of items and bins
    I = [i for i in range(0,len(bin_packing_instance))] # set of items
    B = [i for i in range(0,len(I))] # set of bins -> create as many bins as items because this is the natural upper bound

    # weight and size of items
    w_i = list(bin_packing_instance['Item Weight'])
    s_i = list(bin_packing_instance['Item Size'])
    
    # bin capacity for weight and size
    bin_weight_capacity = list(bin_packing_instance['Bin Weight'])[0]
    bin_size_capacity = list(bin_packing_instance['Bin Size'])[0]
    
    
    # Create a new model in gurobipy
    model = gp.Model('Bin_Packing_with_size_and_weight')
    
    # Decision variables: y is bin selection variable and z is item to bin assignment variable
    y = model.addVars(B, vtype=GRB.BINARY, name='y') # y[b] is 1 if bin b is used
    z = model.addVars(I,B, vtype=GRB.BINARY, name='z') # z[i, b] is 1 if item i is assigned to bin b
    
    
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
    model.Params.Presolve = 2 # Aggressive presolve
    model.Params.MIPGap = 0.01 # gaps allowed because of specific problem structure, gap of e.g. 1% already means exactly solved
    
    # Optimize the model
    model.optimize()
    
    # @Dennis: Hab hier noch nen Textoutput falls du an dem Code rumschraubst und Textoutputs haben willst. Habs jetzt mal auskommentiert drin gelassen
    # Print objective value
    #print(f'Objective value (number of bins used): {model.objVal}')
    
    
    # print unsed bins
    #opened_bins = [b for b in B if y[b].X > 0.5]
    #print(f'Opened bins: {opened_bins}')
    
    
    # print assignment of items to bins
    #for b in B:
     #   assigned_items = [i for i in I if z[i, b].X > 0.5]
      #  if assigned_items:
           # print(f'Bin {b} contains items: {assigned_items}')

    
    # Only return objective value, -> could maybe be adapted if features like utilization or similar things are required
    return model.objVal

# Function to get a list of all possible summations of a given list
def fun_all_sums(values):
    all_sums = []
    num_remaining_items = len(values)
    
    # Create all possible subsets of the list remaining_values
    for i in range(1, num_remaining_items + 1):
        for subset in itertools.combinations(values, i):
            # Get the sum of each subset to compute the total weight/size of the subset
            subset_sum = sum(subset)
            all_sums.append(subset_sum)
    
    return all_sums