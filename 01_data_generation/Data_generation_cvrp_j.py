# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 00:36:02 2023

@author: WWA882
"""


# anbei der Code fürs CVRP.
# Ist glaub bisschen unordentlicher als beim TSP. Aber die Unterschiede 
# sind eigentlich nur die SolveCVRP Funktionen und die spezifischen Features, die sind ganz unten im Code.
# Vielleicht fällt dir auch noch irgendein spezifisches Feature ein.



import random 
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import gurobipy as gp
from gurobipy import GRB
import itertools
from scipy.stats import skew
import matplotlib as plt
import matplotlib.pyplot as plt
import math
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import warnings

# Use the warnings.filterwarnings() function to filter out specific warnings
# For example, to filter out all warnings, you can use the following line:
warnings.filterwarnings("ignore")


###############################################################################
# Generate random instance - settings
###############################################################################
number_customers = [6,7,8,9]

start_instance_id = 0
number_instances = 100

# Define the range for X and Y coordinates
x_range = (0, 100)
y_range = (0, 100)

capacity_min = 10

capacity_max = 18

demand_min = 1

demand_max = 5



###############################################################################
# help functions
###############################################################################


def generate_random_cvrp_instance(num_customers, x_range, y_range, demand_min, demand_max, capacity_min, capacity_max):
    # Create a DataFrame to store customer locations
    columns = ['X', 'Y', "Demand","Vehicle Capacity"]
    customer_locations = pd.DataFrame(columns=columns)
    capacity = random.randint(capacity_min, capacity_max)
    # Generate random X and Y coordinates for each customer
    for _ in range(num_customers):
        x = random.uniform(x_range[0], x_range[1])
        y = random.uniform(y_range[0], y_range[1])
        demand = random.randint(demand_min, demand_max)
        
        # Create a new DataFrame from the new data
        new_row = pd.DataFrame({'X': [x], 'Y': [y], 'Demand': [demand], "Vehicle Capacity": [capacity]})
        
        # Concatenate the new DataFrame with the existing DataFrame
        customer_locations = pd.concat([customer_locations, new_row], ignore_index=True)


    # Add the depot at (5, 5)
    depot = pd.DataFrame({'X': [random.uniform(x_range[0], x_range[1])], 'Y': [random.uniform(y_range[0], y_range[1])], "Demand": 0, "Vehicle Capacity":capacity})
    customer_locations = pd.concat([depot, customer_locations])

    
    return customer_locations

# Function to solve TSP for each cluster
def solve_tsp_for_clusters(df):
    cluster_solutions = {}
    unique_clusters = df['cluster'].unique()

    for cluster in unique_clusters:
        cluster_df = df[df['cluster'] == cluster]

        if cluster_df.shape[0] > 1:  # Skip clusters with only one point (e.g., depot)
            coordinates = cluster_df[['X', 'Y']].values.tolist()
            tsp_solution, tsp_cost = solve_tsp(coordinates)
            cluster_solutions[cluster] = tsp_cost

    return cluster_solutions

def solve_cvrp_shapley(coordinates, demands, capacity):
    num_nodes = len(coordinates)

    # Create a Gurobi model
    model = gp.Model("CVRP")

    # Decision variables
    x = {}
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                x[i, j] = model.addVar(vtype=gp.GRB.BINARY, name=f'x_{i}_{j}')

    # Vehicle flow variables
    u = {}
    for i in range(1, num_nodes):
        u[i] = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name=f'u_{i}')

    model.update()

    # Objective function (minimize total distance)
    model.setObjective(
        gp.quicksum(
            ((coordinates[i][0] - coordinates[j][0]) ** 2 + (coordinates[i][1] - coordinates[j][1]) ** 2) ** 0.5 * x[i, j]
            for i in range(num_nodes) for j in range(num_nodes) if i != j
        ),
        gp.GRB.MINIMIZE
    )

    # Constraints

    # Each customer is visited exactly once
    for i in range(1, num_nodes):
        model.addConstr(gp.quicksum(x[i, j] for j in range(num_nodes) if i != j) == 1, name=f'visit_{i}')

    # Each customer is left exactly once
    for j in range(1, num_nodes):
        model.addConstr(gp.quicksum(x[i, j] for i in range(num_nodes) if i != j) == 1, name=f'leave_{j}')

    # Capacity constraint
    for i in range(1, num_nodes):
        model.addConstr(u[i] >= demands[i], name=f'capacity_{i}')
        model.addConstr(u[i] <= capacity, name=f'capacity_{i}')

    # Subtour elimination constraints
    for i in range(1, num_nodes):
        for j in range(1, num_nodes):
            if i != j:
                model.addConstr(
                    u[i] - u[j] + capacity * x[i, j] <= capacity - demands[j],
                    name=f'subtour_{i}_{j}'
            )
    model.Params.OutputFlag = 0
    model.Params.Presolve = 2  # Aggressive presolve
    model.Params.LazyConstraints = 1
    model.Params.MIPGap = 0.001

    model.optimize()

    if model.status == gp.GRB.OPTIMAL:

        total_cost = model.objVal
        return total_cost
    else:
        return None, None
    
    
def solve_cvrp(coordinates, demands, capacity):
    num_nodes = len(coordinates)

    # Create a Gurobi model
    model = gp.Model("CVRP")

    # Decision variables
    x = {}
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                x[i, j] = model.addVar(vtype=gp.GRB.BINARY, name=f'x_{i}_{j}')

    # Vehicle flow variables
    u = {}
    for i in range(1, num_nodes):
        u[i] = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name=f'u_{i}')

    model.update()

    # Objective function (minimize total distance)
    model.setObjective(
        gp.quicksum(
            ((coordinates[i][0] - coordinates[j][0]) ** 2 + (coordinates[i][1] - coordinates[j][1]) ** 2) ** 0.5 * x[i, j]
            for i in range(num_nodes) for j in range(num_nodes) if i != j
        ),
        gp.GRB.MINIMIZE
    )

    # Constraints

    # Each customer is visited exactly once
    for i in range(1, num_nodes):
        model.addConstr(gp.quicksum(x[i, j] for j in range(num_nodes) if i != j) == 1, name=f'visit_{i}')

    # Each customer is left exactly once
    for j in range(1, num_nodes):
        model.addConstr(gp.quicksum(x[i, j] for i in range(num_nodes) if i != j) == 1, name=f'leave_{j}')

    # Capacity constraint
    for i in range(1, num_nodes):
        model.addConstr(u[i] >= demands[i], name=f'capacity_{i}')
        model.addConstr(u[i] <= capacity, name=f'capacity_{i}')

    # Subtour elimination constraints
    for i in range(1, num_nodes):
        for j in range(1, num_nodes):
            if i != j:
                model.addConstr(
                    u[i] - u[j] + capacity * x[i, j] <= capacity - demands[j],
                    name=f'subtour_{i}_{j}'
            )
    model.Params.OutputFlag = 0
    model.Params.Presolve = 2  # Aggressive presolve
    model.Params.LazyConstraints = 1
    model.Params.MIPGap = 0.001

    model.optimize()

    if model.status == gp.GRB.OPTIMAL:
        routes = []
        for i in range(1, num_nodes):
            if x[0, i].x > 0.5:
                #print("Route starts at customer", i)
                sequence = [0,i]
                
                # Search for the next customer, always save the sequence, and ensure it's different from the previous one
                while sequence[-1] != 0:
                    next_customer = [j for j in range(num_nodes) if j != sequence[-1] and x[sequence[-1], j].x > 0.5][0]
                    sequence.append(next_customer)
                
                routes.append(sequence)
                


        total_cost = model.objVal
        return routes, total_cost
    else:
        return None, None
                
                
                






# =============================================================================
# def visualize_cvrp(coordinates, vehicle_routes):
#     # Plot the depot and customer locations
#     plt.scatter(*zip(*coordinates), marker='o', color='b', label='Customers')
#     plt.scatter(coordinates[0][0], coordinates[0][1], marker='s', color='r', label='Depot')
# 
#     # Plot the routes
#     for route in vehicle_routes:
#         route_adapted = route
#         route_coords = [coordinates[i] for i in route_adapted]
#         route_coords.append(coordinates[0])  # Return to the depot
#         xs, ys = zip(*route_coords)
#         plt.plot(xs, ys, marker='o')
# 
#     plt.legend()
#     plt.xlabel('X Coordinate')
#     plt.ylabel('Y Coordinate')
#     plt.title('CVRP Solution')
#     plt.show()
# =============================================================================


    


def shapley_value(player_index, characteristic_function):
    n = max(max(coalition) for coalition in characteristic_function.keys())
    if player_index < 1 or player_index > n:
        raise ValueError("Player index is out of bounds")
    
    shapley_value = 0.0
    players = list(range(1, n + 1))
    
    for coalition_size in range(1, n + 1):
        for coalition in itertools.combinations(players, coalition_size):
            if player_index in coalition:
                coalition_value = characteristic_function.get(coalition, 0)
                coalition_without_i = set(coalition) - {player_index}
                prev_coalition_value = characteristic_function.get(tuple(sorted(coalition_without_i)), 0)
                marginal_contribution = coalition_value - prev_coalition_value
                
                num_possible_orders = (
                    math.factorial(coalition_size - 1) * math.factorial(n - coalition_size)
                )
                shapley_value += marginal_contribution * (num_possible_orders / math.factorial(n))
    
    return shapley_value

def solve_tsp(coordinates):
    # Create a Gurobi model
    model = gp.Model("TSP")

    # Get the number of coordinates
    num_coordinates = len(coordinates)

    # Create decision variables for the binary variables x_ij
    x = {}
    for i in range(num_coordinates):
        for j in range(num_coordinates):
            if i != j:
                x[i, j] = model.addVar(vtype=gp.GRB.BINARY, name=f'x_{i}_{j}')

    # Define the objective function (minimize the total distance)
    model.setObjective(
        gp.quicksum(((coordinates[i][0] - coordinates[j][0]) ** 2 + (coordinates[i][1] - coordinates[j][1]) ** 2) ** 0.5 * x[i, j]
                     for i in range(num_coordinates) for j in range(num_coordinates) if i != j),
        gp.GRB.MINIMIZE
    )

    # Add constraints to ensure that each coordinate is visited exactly once
    for i in range(num_coordinates):
        model.addConstr(gp.quicksum(x[i, j] for j in range(num_coordinates) if i != j) == 1, name=f'visit_{i}')

    # Add constraints to ensure that each coordinate is left exactly once
    for j in range(num_coordinates):
        model.addConstr(gp.quicksum(x[i, j] for i in range(num_coordinates) if i != j) == 1, name=f'leave_{j}')

    # Add subtour elimination constraints based on cardinality
    for size in range(2, num_coordinates):
        for subset in itertools.combinations(range(num_coordinates), size):
            model.addConstr(gp.quicksum(x[i, j] for i in subset for j in subset if i != j) <= size - 1)

    # Optimize the model
    model.Params.OutputFlag = 0
    model.Params.Presolve = 2  # Aggressive presolve
    model.Params.LazyConstraints = 1
    model.Params.MIPGap = 0.001
    model.optimize()

    # Extract the solution
    if model.status == gp.GRB.OPTIMAL:
        solution = [(i, j) for i in range(num_coordinates) for j in range(num_coordinates) if i != j and x[i, j].x > 0.5]
        total_cost = model.objVal
        return solution, total_cost
    else:
        return None, None

def calculate_cluster_area(cluster_df):
    min_x = cluster_df['X'].min()
    max_x = cluster_df['X'].max()
    min_y = cluster_df['Y'].min()
    max_y = cluster_df['Y'].max()
    
    # Calculate the side length of the square
    side_length = max(max_x - min_x, max_y - min_y)
    
    # Calculate the area of the square
    cluster_area = side_length * side_length
    
    return cluster_area


# Function to calculate the average distance to the 3 closest customers
def average_distance_to_closest_customers(row):
    distances = customer_distances[row.name - 1]
    closest_indices = distances.argsort()[:3]
    closest_distances = distances[closest_indices]
    return closest_distances.mean()

# Function to calculate the distance between two coordinates
def distance(coord1, coord2):
    return ((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2) ** 0.5






###############################################################################
# Start generating
###############################################################################
for n_customers in number_customers:
    
    instance_id = start_instance_id
    while instance_id < start_instance_id + number_instances:
        
        
        
        
        num_customers = n_customers
        
        
        
        tsp_instance = generate_random_cvrp_instance(num_customers, x_range, y_range, demand_min, demand_max, capacity_min, capacity_max)
        
        
        
        # Calculate the distance between the depot and each location
        x_coord = list(tsp_instance["X"])[0]
        y_coord = list(tsp_instance["Y"])[0]
        
        tsp_instance['Distance_to_Depot'] = ((tsp_instance['X'] - x_coord) ** 2 + (tsp_instance['Y'] - y_coord) ** 2) ** 0.5
        
        
        tsp_instance
        
        
        
        # Calculate the average distance to the depot from all customers
        average_distance_to_depot = tsp_instance.loc[1:, 'Distance_to_Depot'].mean()
        
        # Calculate the distance to the depot divided by the average distance for each customer
        tsp_instance['Distance_to_Depot_Ratio'] = tsp_instance['Distance_to_Depot'] / average_distance_to_depot
        
        
        customer_distances = cdist(tsp_instance.iloc[1:][['X', 'Y']], tsp_instance.iloc[1:][['X', 'Y']], metric='euclidean')
        
        
        # Replace diagonal values with a large number (e.g., infinity)
        np.fill_diagonal(customer_distances, np.inf)
        
        # Initialize lists to store distances to the closest customers
        closest_distances = []
        second_closest_distances = []
        third_closest_distances = []
        fourth_closest_distances = []
        
        # Loop through each customer (excluding the depot)
        for i in range(0, len(customer_distances)):
            # Sort the distances for the current customer
            sorted_distances = np.sort(customer_distances[i])
            
            # Get the closest, second closest, third closest, and fourth closest distances
            closest = sorted_distances[0]
            second_closest = sorted_distances[1]
            third_closest = sorted_distances[2]
            fourth_closest = sorted_distances[3]
            
            # Append the distances to the respective lists
            closest_distances.append(closest)
            second_closest_distances.append(second_closest)
            third_closest_distances.append(third_closest)
            fourth_closest_distances.append(fourth_closest)
        
        # Add columns for the distances to the closest, second closest, third closest, and fourth closest customers
        tsp_instance['Closest_Customer_Distance'] = [np.nan] + closest_distances
        tsp_instance['Second_Closest_Customer_Distance'] = [np.nan] + second_closest_distances
        tsp_instance['Third_Closest_Customer_Distance'] = [np.nan] + third_closest_distances
        tsp_instance['Fourth_Closest_Customer_Distance'] = [np.nan] + fourth_closest_distances
        
        
        
        tsp_instance
        
        
        ###############################################################################
        # Mean distance to all other customers
        ###############################################################################
        # Calculate distances from each customer to all other customers (excluding depot)
        num_customers = len(tsp_instance) - 1
        distances = np.zeros((num_customers, num_customers))
        for i in range(1, num_customers + 1):
            for j in range(1, num_customers + 1):
                if i != j:
                    distances[i - 1, j - 1] = np.linalg.norm(tsp_instance.iloc[i][["X", "Y"]] - tsp_instance.iloc[j][["X", "Y"]])
        
        # Calculate the mean distance to other customers (excluding depot)
        mean_distances = distances.mean(axis=1)
        
        # Add the "Mean distance to other customers" column to the DataFrame
        # We add a NaN value for the depot since it doesn't have a mean distance
        tsp_instance["Mean distance to other customers"] = [np.nan] + mean_distances.tolist()
        
        
        
        # Reset the index
        tsp_instance = tsp_instance.reset_index(drop=True)
        
        
        
        
        
        
        ###############################################################################
        # Add solution specific features
        ###############################################################################
        
        # Determine marginal costs
        # Create a copy of the original tsp_instance DataFrame
        tsp_instance_with_customer = tsp_instance.copy()
        
        
        tsp_instance.columns
        
        coord = [(tsp_instance_with_customer["X"][i],tsp_instance_with_customer["Y"][i]) for i in range(len(tsp_instance_with_customer))]
        demands = list(tsp_instance_with_customer["Demand"])
        demands
        capacity = list(tsp_instance_with_customer["Vehicle Capacity"])[0]
        capacity
        
        
        
        
        
        
        # Calculate the TSP cost without the customer
        sequence_with_customer, cost_with_customer = solve_cvrp(coord, demands, capacity)
        
        # Visualize the solution
        #visualize_cvrp(coord, sequence_with_customer)
        
        
        # Initialize an empty list to store the marginal costs
        marginal_costs = []
        
        # Iterate through each customer and calculate the marginal cost
        for i in tsp_instance_with_customer.index[1:]:
        
            # Remove the customer from the instance to calculate the cost without the customer
            tsp_instance_without_customer = tsp_instance_with_customer.drop(i)
            
            tsp_instance_without_customer = tsp_instance_without_customer.reset_index()
            
            # 
            coord = [(tsp_instance_without_customer["X"][j],tsp_instance_without_customer["Y"][j]) for j in range(len(tsp_instance_without_customer))]
            demands = [tsp_instance_without_customer["Demand"][j] for j in range(len(tsp_instance_without_customer))]
            
            
            # Calculate the TSP cost with the customer
            cost_without_customer = solve_cvrp_shapley(coord, demands, capacity)
            
            # Calculate the marginal cost
            marginal_cost = cost_with_customer - cost_without_customer
            
            # Append the marginal cost to the list
            marginal_costs.append(marginal_cost)
        
        
        
        # Add the marginal costs to the DataFrame
        tsp_instance_with_customer['Marginal_Cost'] = [0] + marginal_costs  # Add a dummy value for the depot
        
        
        sum(marginal_costs)
        
        tsp_instance["Marginal_Cost"] = tsp_instance_with_customer["Marginal_Cost"]
        
        
        
        
        
        
        ###############################################################################
        # Add cost savings
        ###############################################################################
        
        sequence_with_customer
        
        
        
        
        coord = [(tsp_instance["X"][i],tsp_instance["Y"][i]) for i in range(len(tsp_instance))]
        
        savings = {}
        
        #for i in range(1,len(tsp_instance)):
        
        
        cost_savings = 999
        node_before = 999
        node_after = 999
        
        for i in range(1,len(tsp_instance)):
            
            #print("Customer: " + str(i))
            
            for route in sequence_with_customer:
                index =0
                for node in route:
                    if node == i:
                        #print("Node before: " + str(route[index-1]))
                        #print("Node after: " + str(route[index+1]))
                        node_before = route[index-1]
                        node_after= route[index+1]
                        
                    index = index+1
                    
            # determine cost savings
            cost_savings = distance(coord[node_before], coord[i]) + distance(coord[i], coord[node_after]) - distance(coord[node_before], coord[node_after])
        
            # save in dictionary
            savings[i] = cost_savings
                    
        
        coord
        savings_list = [0] + [savings[i] for i in range(1,num_customers+1)]
        
        
        tsp_instance["Savings"] = savings_list
        
        tsp_instance
        
        
        
        
        ###############################################################################
        # Calculate Shapley Value
        ###############################################################################
        
        # Calculate Shapley values for each customer
        num_customers = len(tsp_instance) - 1
        shapley_values = [0] * num_customers
        
        
        list_of_all_subsets = []
        
        for customer in range(1, num_customers + 1):
            subsets = itertools.combinations(range(1, num_customers + 1), customer)
            for subset in subsets:
                list_of_all_subsets.append(subset)
        
        
        
        subset_total_cost = {}
        count = 0
        
        
       
        for subset in list_of_all_subsets:
        
            
            coordinates =[(tsp_instance["X"][0], tsp_instance["Y"][0])]  + [(tsp_instance["X"][i], tsp_instance["Y"][i]) for i in subset]
            demands = [tsp_instance["Demand"][i] for i in subset]
            demands = [0] + demands
            
            sequence, total_cost  = solve_cvrp(coordinates, demands, capacity)
            
            couter = count + 1
            
            #print("total cost: " + str(total_cost)) 
            
            subset_total_cost[subset] = total_cost
        
        
        
        shapley_values = [0]
        shapley_sum = 0
        for i in range(1,num_customers+1):
            #print("Customer: " + str(i))
            #print(subset_total_cost)
            shapley_val = shapley_value(i, subset_total_cost)
            #print(shapley_val)
            shapley_sum = shapley_sum + shapley_val
            shapley_values = shapley_values + [shapley_val]
        
        
        #print(shapley_sum)
        
        tsp_instance["Shapley Value"] = shapley_values
        
        
        
        tsp_instance["Total cost"] = [total_cost for i in range(0,len(tsp_instance))]
        
        tsp_instance["Number Customers"] = [num_customers for i in range(0,len(tsp_instance))]
        
        centroid_x = tsp_instance.loc[1:, 'X'].mean()
        centroid_y = tsp_instance.loc[1:, 'Y'].mean()
        
        # Calculate the distance to centroid for each customer (excluding the depot) and save it in a new column
        tsp_instance['Distance_to_gravity_center'] = np.sqrt((tsp_instance.loc[1:, 'X'] - centroid_x) ** 2 + (tsp_instance.loc[1:, 'Y'] - centroid_y) ** 2)
        
        
        
        
        
        
        # Cluster customers
        # Assuming you have x and y coordinates in columns 'x' and 'y'
        data = tsp_instance
        X = data[['X', 'Y']]
        
        
        # Choose the number of clusters (K) - you need to specify this based on your problem
        K = 2  # Change this to the desired number of clusters
        
        K_min=2  
        while True:
            # Initialize and fit the K-Means model
            kmeans = KMeans(n_clusters=K, random_state=0)
            kmeans.fit(X)
            
            # Get the number of unique cluster labels created
            unique_labels = len(np.unique(kmeans.labels_))
            
            if unique_labels >= K_min+1:
                break  # Stop if the minimum number of clusters is achieved
            
            K=K+1
        
        
        
        # Get cluster assignments for each data point
        labels = kmeans.labels_
        
        # Add cluster labels to the original data
        data['cluster'] = labels
        
        distinct_values = data[data['cluster'] != -1]['cluster'].unique()
        
        
        # Exclude the depot (first row) from clustering
        data.loc[0, 'cluster'] = -1  # Assign an unused cluster label to the depot
        
# =============================================================================
#         # Plot the clustered data (for illustration purposes)
#         colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
#         
#         for i in range(K):
#             cluster_data = data[data['cluster'] == i]
#             plt.scatter(cluster_data['X'], cluster_data['Y'], c=colors[i], label=f'Cluster {i + 1}')
#             
#             # Plot cluster centers
#             cluster_centers = kmeans.cluster_centers_
#             plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', marker='x', s=100, label='Cluster Centers')
#             
#         plt.xlabel('X')
#         plt.ylabel('Y')
#         plt.title('K-Means Clustering')
#         plt.legend()
#         plt.show()
# =============================================================================
        
        # Calculate the distances from each customer to its cluster center
        data['distance_to_cluster_center'] = np.nan  # Create a new column to store distances
        distinct_values = data[data['cluster'] != -1]['cluster'].unique()
        for cluster_id in list(distinct_values):
            if cluster_id != -1:  # Exclude the depot
                # Get the scaled cluster center coordinates
                cluster_center = kmeans.cluster_centers_[cluster_id]  # Scaled cluster centers
            
                # Filter data points belonging to the current cluster
                cluster_data = data[data['cluster'] == cluster_id]
            
                # Calculate distances for each customer in the cluster
                distances = np.linalg.norm(cluster_data[['X', 'Y']] - cluster_center, axis=1)
            
                # Assign the calculated distances to the corresponding rows in the DataFrame
                data.loc[data['cluster'] == cluster_id, 'distance_to_cluster_center'] = distances
        
        # Calculate the minimum distance to customers of other clusters
        data['min_distance_to_other_cluster'] = np.nan  # Create a new column for minimum distances
        
        distinct_values = data[data['cluster'] != -1]['cluster'].unique()
        for cluster_id in list(distinct_values):
            if cluster_id != -1:  # Exclude the depot
                # Filter data points belonging to the current cluster
                current_cluster_data = data[data['cluster'] == cluster_id]
            
                # Calculate distances to all customers in other clusters
                distances_to_other_clusters = cdist(current_cluster_data[['X', 'Y']], 
                                                    data[(data['cluster'] != cluster_id) & (data['cluster'] != -1)][['X', 'Y']], 
                                                    metric='euclidean')
            
                # Calculate the minimum distance to customers of other clusters for each customer
                min_distances = np.min(distances_to_other_clusters, axis=1)
            
                # Assign the minimum distances to the corresponding rows in the DataFrame
                data.loc[data['cluster'] == cluster_id, 'min_distance_to_other_cluster'] = min_distances
        
        # Calculate the number of customers in each cluster (excluding the depot)
        cluster_sizes = data[data['cluster'] != -1]['cluster'].value_counts().to_dict()
        
        distinct_values = data[data['cluster'] != -1]['cluster'].unique()
        for cluster_id in list(distinct_values):
            if cluster_id != -1:  # Exclude the depot
                # Filter data points belonging to the current cluster (excluding the depot)
                current_cluster_data = data[(data['cluster'] == cluster_id) & (data['cluster'] != -1)]
            
                # Calculate distances to all customers in other clusters (excluding the depot)
                distances_to_other_clusters = cdist(current_cluster_data[['X', 'Y']], 
                                                    data[(data['cluster'] != cluster_id) & (data['cluster'] != -1)][['X', 'Y']], 
                                                    metric='euclidean')
            
                # Calculate the minimum distance to customers of other clusters for each customer
                min_distances = np.min(distances_to_other_clusters, axis=1)
            
                # Assign the minimum distances to the corresponding rows in the DataFrame
                data.loc[(data['cluster'] == cluster_id) & (data['cluster'] != -1), 'min_distance_to_other_cluster'] = min_distances
            
                # Add the number of customers in the same cluster (excluding the depot) to each row
                data.loc[(data['cluster'] == cluster_id) & (data['cluster'] != -1), 'customers_in_same_cluster'] = cluster_sizes[cluster_id]
        
        
        
        depot_coordinates = data.loc[0, ['X', 'Y']]
        data['cluster_distance_to_depot'] = np.nan  # Create a new column to store distances
        
        distinct_values = data[data['cluster'] != -1]['cluster'].unique()
        for cluster_id in list(distinct_values):
            if cluster_id != -1:  # Exclude the depot
                # Get the scaled cluster center coordinates
                cluster_center = kmeans.cluster_centers_[cluster_id]  # Scaled cluster centers
            
            
                # Calculate the distance from the cluster center to the depot
                distance_to_depot = np.linalg.norm(cluster_center - depot_coordinates)
            
                # Assign the calculated distances to the corresponding rows in the DataFrame
                data.loc[data['cluster'] == cluster_id, 'cluster_distance_to_depot'] = distance_to_depot
        
        tsp_instance = data
        
        
        
        
        # create ratios 
        mean_marginal_costs = np.mean(tsp_instance['Marginal_Cost'])
        tsp_instance['Marginal_Cost_ratio'] = tsp_instance['Marginal_Cost']/ mean_marginal_costs
        
        
        # create ratios 
        mean_savings = np.mean(tsp_instance['Savings'])
        tsp_instance['Savings_ratio'] = tsp_instance['Savings']/ mean_savings
        
        # create ratios 
        mean_Distance_to_gravity_center = np.mean(tsp_instance['Distance_to_gravity_center'])
        tsp_instance['Distance_to_gravity_center_ratio'] = tsp_instance['Distance_to_gravity_center']/ mean_Distance_to_gravity_center
        
        
        # create ratios 
        mean_distance_to_cluster_center = np.mean(tsp_instance['distance_to_cluster_center'])
        tsp_instance['distance_to_cluster_center_ratio'] = tsp_instance['distance_to_cluster_center']/ mean_distance_to_cluster_center
        
        # Calculate the means of the distance columns
        mean_closest_distance = np.mean(tsp_instance['Closest_Customer_Distance'])
        mean_second_closest_distance = np.mean(tsp_instance['Second_Closest_Customer_Distance'])
        mean_third_closest_distance = np.mean(tsp_instance['Third_Closest_Customer_Distance'])
        mean_fourth_closest_distance = np.mean(tsp_instance['Fourth_Closest_Customer_Distance'])
        
        # Create ratio columns for Closest, Second Closest, Third Closest, and Fourth Closest distances
        tsp_instance['Closest_Customer_Distance_Ratio'] = tsp_instance['Closest_Customer_Distance'] / mean_closest_distance
        tsp_instance['Second_Closest_Customer_Distance_Ratio'] = tsp_instance['Second_Closest_Customer_Distance'] / mean_second_closest_distance
        tsp_instance['Third_Closest_Customer_Distance_Ratio'] = tsp_instance['Third_Closest_Customer_Distance'] / mean_third_closest_distance
        tsp_instance['Fourth_Closest_Customer_Distance_Ratio'] = tsp_instance['Fourth_Closest_Customer_Distance'] / mean_fourth_closest_distance
        
        
        # create ratios 
        mean_cluster_distance_to_depot = np.mean(tsp_instance['cluster_distance_to_depot'])
        tsp_instance['cluster_distance_to_depot_ratio'] = tsp_instance['cluster_distance_to_depot']/ mean_cluster_distance_to_depot
        
        
        # create ratios 
        mean_Mean_distance_to_other_customers= np.mean(tsp_instance['Mean distance to other customers'])
        tsp_instance['Mean distance to other customers_ratio'] = tsp_instance['Mean distance to other customers']/ mean_Mean_distance_to_other_customers
        
        
        # create ratios 
        mean_min_distance_to_other_cluster= np.mean(tsp_instance['min_distance_to_other_cluster'])
        tsp_instance['min_distance_to_other_cluster_ratio'] = tsp_instance['min_distance_to_other_cluster']/ mean_min_distance_to_other_cluster
        
        
        
        cluster_areas = tsp_instance.groupby('cluster').apply(calculate_cluster_area).reset_index(name='Cluster Area')
        
        
        # Merge the cluster areas back into the original DataFrame based on the 'Cluster' column
        tsp_instance = tsp_instance.merge(cluster_areas, on='cluster', how='left')
        
        # create ratios 
        mean_cluster_area= np.mean(tsp_instance['Cluster Area'])
        tsp_instance['Cluster Area ratio'] = tsp_instance['Cluster Area']/ mean_cluster_area
        
        num_clusters = tsp_instance.loc[tsp_instance['cluster'] != -1, 'cluster'].nunique()
        
        # Create a new column 'Number Clusters' with the same value in each row
        tsp_instance['Number Clusters'] = num_clusters
        
        tsp_instance
        
        
        
        # Solve TSP for each cluster
        cluster_solutions = solve_tsp_for_clusters(tsp_instance)
        
        # Add a new column "Cluster TSP" to the original DataFrame
        tsp_instance['Cluster TSP'] = tsp_instance['cluster'].map(cluster_solutions).fillna(-1)
        
        
        tsp_instance
        
        
        # Determine shapley value for each cluster
        
        
        # Extract depot and cluster centroids
        depot = tsp_instance.iloc[0]
        clustered_customers = tsp_instance[tsp_instance["cluster"] >= 0]
        
        # Calculate cluster centroids
        cluster_centroids = clustered_customers.groupby("cluster")[["X", "Y"]].mean().reset_index()
        
        # Combine depot and cluster centroids
        coordinates = [(depot["X"], depot["Y"])] + list(zip(cluster_centroids["X"], cluster_centroids["Y"]))
        
        # Solve TSP for cluster centroids
        solution, total_cost = solve_tsp(coordinates)
        
        # Update the DataFrame with the total cost in the "TSP to cluster centroids" column
        if solution is not None:
            tsp_instance["TSP to cluster centroids"] = total_cost
        else:
            tsp_instance["TSP to cluster centroids"] = np.nan
        
        
        
        
                    
        
        
        
        # Depot coordinates
        # Extract depot coordinates from the first row
        depot_x = tsp_instance.loc[0, "X"]
        depot_y = tsp_instance.loc[0, "Y"]
        
        # Add 'X_Depot' and 'Y_Depot' columns with depot coordinates
        tsp_instance['X_Depot'] = depot_x
        tsp_instance['Y_Depot'] = depot_y
        
        # Calculate the mean of X and insert it into the "X_mean" column for all rows
        tsp_instance['X_mean'] = tsp_instance.iloc[1:]['X'].mean()
        
        # Calculate the standard deviation of X and insert it into the "X_std" column for all rows
        tsp_instance['X_std'] = tsp_instance.iloc[1:]['X'].std()
        
        # Calculate the maximum value of X and insert it into the "X_max" column for all rows
        tsp_instance['X_max'] = tsp_instance.iloc[1:]['X'].max()
        
        # Calculate the minimum value of X and insert it into the "X_min" column for all rows
        tsp_instance['X_min'] = tsp_instance.iloc[1:]['X'].min()
        
        # Calculate the mean of Y and insert it into the "Y_mean" column for all rows
        tsp_instance['Y_mean'] = tsp_instance.iloc[1:]['Y'].mean()
        
        # Calculate the standard deviation of Y and insert it into the "Y_std" column for all rows
        tsp_instance['Y_std'] = tsp_instance.iloc[1:]['Y'].std()
        
        # Calculate the maximum value of Y and insert it into the "Y_max" column for all rows
        tsp_instance['Y_max'] = tsp_instance.iloc[1:]['Y'].max()
        
        # Calculate the minimum value of Y and insert it into the "Y_min" column for all rows
        tsp_instance['Y_min'] = tsp_instance.iloc[1:]['Y'].min()
            
        # Extract X and Y values for the entire column, excluding the first row (depot)
        x_values = tsp_instance['X'].iloc[1:]
        y_values = tsp_instance['Y'].iloc[1:]
        
        # Calculate the correlation between X and Y values
        correlation = np.corrcoef(x_values, y_values)[0, 1]
        
        # Calculate the skewness for X and Y values
        skewness_x = skew(x_values)
        skewness_y = skew(y_values)
        
        # Add the calculated values to each row in their respective columns
        tsp_instance['Correlation'] = correlation
        tsp_instance['Skewness_X'] = skewness_x
        tsp_instance['Skewness_Y'] = skewness_y
        
        tsp_instance["Instance_id"] = instance_id
        
        
        # Fill the NaN values in the first row (depot row) with -1
        tsp_instance.fillna(-1, inplace=True)
        
        
        tsp_instance
        
        
        ###############################################################################
        # Add CVRP specific features
        ###############################################################################
        
        
        total_demand_volume = tsp_instance["Demand"].sum()
        tsp_instance["Total demand"] = total_demand_volume
        
        # Demand ratio
        number_customers = len(tsp_instance)-1
        tsp_instance["Demand ratio"] = tsp_instance["Demand"] / (total_demand_volume /number_customers)
        
        
        tsp_instance.columns
        # Cluster demand
        # Calculate the sum of Demands for each cluster and add it to a new column
        tsp_instance['Cluster demand'] = tsp_instance.groupby('cluster')['Demand'].transform('sum')
        
        
        tsp_instance['Cluster demand'][0] = -1
        # Customer cluster demand - ratio 
        tsp_instance['Customer Cluster demand ratio'] =  tsp_instance['Demand'] / (tsp_instance['Cluster demand'] / tsp_instance['customers_in_same_cluster']) 
        
        
        # Cluster demand - ratio
        tsp_instance['Cluster demand ratio'] = tsp_instance['Cluster demand'] / (tsp_instance['Total demand']/tsp_instance['Number Clusters'])
        
        
        
        
        tsp_instance.to_excel("C:/Users/JPFontaine/Documents/Johannes/Shapley_Approximation/CVRP_Code/Instanzen/"+str(n_customers) + "_Customer_instance" + str(instance_id) + ".xlsx")
        instance_id = instance_id+1













