# -*- coding: utf-8 -*-
"""
Author: Johannes Gückel
Topic: Generating and preparing TSP Instances for Shapley Value analsis
"""


###############################################################################
# Import packages
###############################################################################
import random 
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import gurobipy as gp
from gurobipy import GRB
import itertools
from scipy.stats import skew
import math
import itertools
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings


###############################################################################
# Generate random instance
###############################################################################

num_customer_list = [10,11] # <- Hier schreibst die Anzahl der Kunden pro instanz rein
start_value=0 # <- Der Wert hier wird als Instanz_id reingeschrieben, hab das drin weil ich immer mehrere Skripte Parallel laufen lasse, da können nicht alle bei 0 starten
numbe_of_instances_per_size =10    # <- Diese Einstellungen generieren jetzt je 10 Instanzen mit 10 und 11 Kunden


###############################################################################
# Functions for TSP solving & Shapley Value
###############################################################################

def generate_random_tsp_instance(num_customers, x_range, y_range):
    """
    

    Parameters
    ----------
    num_customers : int
        number of customers in generated instance
    x_range : range(int,int)
        range der x Koordinate -> bisher immer zwischen 0 und 100 gesetzt
    y_range : range(int,int)
        range der x Koordinate -> bisher immer zwischen 0 und 100 gesetzt

    Returns
    -------
    customer_locations : Dataframe
        dataframe of coordinates

    """
    # Create a list to store customer locations as dictionaries
    customer_data = []

    # Generate random X and Y coordinates for each customer
    for _ in range(num_customers):
        x = random.uniform(x_range[0], x_range[1])
        y = random.uniform(y_range[0], y_range[1])
        customer_data.append({'X': x, 'Y': y})

    # Generate random X and Y coordinates for the depot
    depot_x = random.uniform(x_range[0], x_range[1])
    depot_y = random.uniform(y_range[0], y_range[1])
    depot_data = {'X': depot_x, 'Y': depot_y}

    # Insert the depot data at the beginning of the list
    customer_data.insert(0, depot_data)

    # Create a DataFrame from the list of customer locations
    customer_locations = pd.DataFrame(customer_data)

    return customer_locations




def shapley_value(player_index, characteristic_function):
    """
    Parameters
    ----------
    player_index : int
        Index des Kunden.
    characteristic_function : dictionary
        characteristic function of the tsp instance -> see later how this is structured

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    shapley_value : double
        shapley value of the player

    """
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


def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)
   


# Die Funktion wird später zur Berechnung der Shapley Values für ein Cluster verwendet -> glaub ich hab die Shapley Values für Cluster eh wieder rausgenommen aus den Features, von daher eher weniger relevant. Aber kannst dir mal anschauen falls du da doch noch ne Idee dazu  hast.
def shapley_value_coalition(characteristic_function):
    shapley_values = {}
    total_players = set()

    # Add the empty coalition with cost 0
    characteristic_function[()] = 0.0

    # Determine the total set of players
    for coalition in characteristic_function.keys():
        total_players.update(set(coalition))

    num_players = len(total_players)

    for player in total_players:
        shapley_values[player] = 0.0

    for coalition, value in characteristic_function.items():
        coalition_size = len(coalition)
        for player in coalition:
            subcoalition = tuple(member for member in coalition if member != player)
            marginal_contribution = value - characteristic_function[subcoalition]
            shapley_values[player] += (marginal_contribution * (factorial(coalition_size - 1) * factorial(num_players - coalition_size)))/factorial(num_players)
            
            #print("Number players " + str(num_players))
            #print("Marginal contribution of player " + str(player) + "  is " + str(marginal_contribution))
    return shapley_values



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

    # Optimize the model -> set Gurobi parameters to solve as fast as possible
    model.Params.OutputFlag = 0
    model.Params.Presolve = 2  # Aggressive presolve
    model.Params.LazyConstraints = 1
    model.Params.MIPGap = 0.0001 # stop after Gap smaller than 0.01 %
    model.optimize()

    # Extract the solution
    if model.status == gp.GRB.OPTIMAL:
        solution = [(i, j) for i in range(num_coordinates) for j in range(num_coordinates) if i != j and x[i, j].x > 0.5]
        total_cost = model.objVal
        return solution, total_cost
    else:
        return None, None
    


# Function to solve TSP for each cluster -> Wie gesagt, hab das am Ende eh wieder aus den Features rausgenommen, von daher nicht so relevant aktuell
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


# Function to calculate the distance between two coordinates
def distance(coord1, coord2):
    return ((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2) ** 0.5





###############################################################################
# Start generating
###############################################################################

# Define the range for X and Y coordinates
x_range = (0, 100)
y_range = (0, 100)



for num_customers in num_customer_list:
    instance_id = 100000* num_customers * start_value # -> Random gesetzt, hauptsache es gibt bei Parallelisierung keine Überschneidung der Indizes
    instance_numbers = range(start_value,start_value + numbe_of_instances_per_size)
    
    for inumber in instance_numbers:
        
        
        # Generate instaqnce
        tsp_instance = generate_random_tsp_instance(num_customers, x_range, y_range)
        
        
        # Calculate the distance between the depot and each location
        x_coord = list(tsp_instance["X"])[0]
        y_coord = list(tsp_instance["Y"])[0]
        tsp_instance['Distance_to_Depot'] = ((tsp_instance['X'] - x_coord) ** 2 + (tsp_instance['Y'] - y_coord) ** 2) ** 0.5
        
        
        # Calculate the average distance to the depot from all customers
        average_distance_to_depot = tsp_instance.loc[1:, 'Distance_to_Depot'].mean()
        
        # Calculate the distance to the depot divided by the average distance for each customer
        tsp_instance['Distance_to_Depot_Ratio'] = tsp_instance['Distance_to_Depot'] / average_distance_to_depot

        tsp_instance['X'] = pd.to_numeric(tsp_instance['X'])
        tsp_instance['Y'] = pd.to_numeric(tsp_instance['Y'])

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
        
        tsp_instance
        
        coord = [(tsp_instance["X"][i],tsp_instance["Y"][i]) for i in range(len(tsp_instance))]
        
        coord
        sequence, total_cost  = solve_tsp(coord)
        
        
        ###############################################################################
        # Add solution specific features
        ###############################################################################
        
        # Determine marginal costs
        # Create a copy of the original tsp_instance DataFrame
        tsp_instance_with_customer = tsp_instance.copy()
        
        # 
        coord = [(tsp_instance_with_customer["X"][i],tsp_instance_with_customer["Y"][i]) for i in range(len(tsp_instance_with_customer))]
        
        # Calculate the TSP cost without the customer
        sequence_with_customer, cost_with_customer = solve_tsp(coord)
        
        # Initialize an empty list to store the marginal costs
        marginal_costs = []
        
        # Iterate through each customer and calculate the marginal cost
        for i in tsp_instance_with_customer.index[1:]:
            
            # Remove the customer from the instance to calculate the cost without the customer
            tsp_instance_without_customer = tsp_instance_with_customer.drop(i)
            
            tsp_instance_without_customer = tsp_instance_without_customer.reset_index()
            
            # 
            coord = [(tsp_instance_without_customer["X"][j],tsp_instance_without_customer["Y"][j]) for j in range(len(tsp_instance_without_customer))]
            
            # Calculate the TSP cost with the customer -> Das könnte ich effizienter machen, später wird die charakteristische Funktion ja eh berechnet, von daher ist der Schritt eigentlich doppelt
            sequence_without_customer, cost_without_customer = solve_tsp(coord)
            
            # Calculate the marginal cost
            marginal_cost = cost_with_customer - cost_without_customer
            
            # Append the marginal cost to the list
            marginal_costs.append(marginal_cost)
        
        # Add the marginal costs to the DataFrame
        tsp_instance_with_customer['Marginal_Cost'] = [0] + marginal_costs  # Add a dummy value for the depot
        
        tsp_instance_with_customer["Marginal_Cost"].sum()
        
        tsp_instance_with_customer
        
        tsp_instance["Marginal_Cost"] = tsp_instance_with_customer["Marginal_Cost"]

        ###############################################################################
        # Add cost savings
        ###############################################################################
        
        used_arcs = sequence

        coord = [(tsp_instance["X"][i],tsp_instance["Y"][i]) for i in range(len(tsp_instance))]
        
        savings = {}
        
        for i in range(1,len(tsp_instance)):
            
            #print(i)
            cost_savings = 999
            
            node_before = 999
            node_after = 999
            
            # determine node before
            for arc in used_arcs:
                if arc[1]==i:
                    node_before = arc[0]
            
            # determine node afterwards
            for arc in used_arcs:
                if arc[0]==i:
                    node_after = arc[1]
           
            # determine cost savings
            cost_savings = distance(coord[node_before], coord[i]) + distance(coord[i], coord[node_after]) - distance(coord[node_before], coord[node_after])
            
            #print("Here i am")
            
            # save in dictionary
            savings[i] = cost_savings

        savings
        
        savings_list = [0] + [savings[i] for i in range(1,num_customers+1)]
        savings_list
        
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
        
        subset_total_cost = {} # Hier berechne ich ertst die charakteristische Funlktion und darauf aufbauen dann den Shapley Value
        count = 0
        for subset in list_of_all_subsets:
            
            coordinates =[(tsp_instance["X"][0], tsp_instance["Y"][0])]  + [(tsp_instance["X"][i], tsp_instance["Y"][i]) for i in subset]
            
            sequence, total_cost  = solve_tsp(coordinates)
            
            couter = count + 1
            
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
            
        data = tsp_instance

        # Assuming you have x and y coordinates in columns 'X' and 'Y'
        X = data[['X', 'Y']]
 
        # -> Hier kannstz du K-MEans mit Silhouette berechnen, hatte das bisher immer relativ unsauber gemacht :D. Wichtig wäre aber ne obergrenze. Also nicht, dass es am Ende nur mega kleine Cluster mit 1-2 Kunden gibt
        # Choose the number of clusters (K) - you need to specify this based on your problem
        K = 3  # Change this to the desired number of clusters
        
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
        
        
        tsp_instance.columns
        
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
        
        # Create a dictionary to store characteristic functions for Shapley value calculation
        characteristic_function = {}
        
        # Initialize Shapley values
        shapley_values = [0] * len(cluster_centroids)
        
        # Create a mapping of cluster identifiers to unique integer indices
        cluster_to_index = {cluster: index for index, cluster in enumerate(cluster_centroids["cluster"])}
               
        # Calculate Shapley values for each cluster centroid
        for i in range(len(cluster_centroids)):
            cluster_identifier = cluster_centroids["cluster"].iloc[i]
            player_index = cluster_to_index[cluster_identifier]
            
            # Generate all combinations of clusters (excluding the depot)
            all_clusters = cluster_centroids["cluster"].tolist()
            for coalition_size in range(1, len(all_clusters) + 1):
                for coalition in itertools.combinations(all_clusters, coalition_size):
                    if cluster_identifier in coalition:
                        # Solve TSP for the current coalition
                        coalition_coordinates = [coordinates[all_clusters.index(c)+1] for c in coalition]
                        solution, coalition_cost = solve_tsp([(tsp_instance["X"][0], tsp_instance["Y"][0])] + coalition_coordinates)
                        
                        # Update the characteristic function with coalition cost
                        coalition_tuple = tuple(coalition)
                        characteristic_function[coalition_tuple] = coalition_cost
        
        
        
        shapley_values = shapley_value_coalition(characteristic_function)
        #print("Shapley Values:", shapley_values)
        
        shapley_values
        
        tsp_instance["Shapley Value Cluster"] = tsp_instance["cluster"].map(shapley_values)
        
        
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
        
        instance_id = instance_id+1
        # Fill the NaN values in the first row (depot row) with -1
        tsp_instance.fillna(-1, inplace=True)
        
        # Save file
        tsp_instance.to_excel("C:/Users/WWA882/Documents/Wissenschaftlicher_Mitarbeiter/Forschung/Paper_3/Code/"+str(num_customers) + "_Customer_instance" + str(inumber) + ".xlsx")
        




