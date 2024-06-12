###############################################################################
# Import packages
###############################################################################
import os
import time
import random 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mglearn
from scipy.spatial.distance import cdist
import gurobipy as gp
from gurobipy import GRB
from scipy.stats import skew
import math
import itertools
from sklearn.cluster import DBSCAN

###############################################################################
# Functions for generating instances and loading or saving a file
###############################################################################
# Function to generate a TSP instance
def generate_tsp_instance(num_customers, x_range, y_range):
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
        customer_data.append({'X': x, 'Y': y}) # List with dictionaries. One for each customer with X and Y coordinate

    # Generate random X and Y coordinates for the depot
    x_depot = random.uniform(x_range[0], x_range[1])
    y_depot = random.uniform(y_range[0], y_range[1])
    depot_data = {'X': x_depot, 'Y': y_depot}

    # Insert the depot data at the beginning of the list
    customer_data.insert(0, depot_data)

    # Create a DataFrame from the list of customer locations
    customer_locations = pd.DataFrame(customer_data) # One row for each customer with X and Y coordinate. First row is the depot

    return customer_locations

# Function to generate a CVRP instance
def generate_cvrp_instance(num_customers, x_range, y_range, demand_min, demand_max, capacity_min, capacity_max):
    # Create a DataFrame to store customer locations
    columns = ['X', 'Y', 'Vehicle Capacity', 'Demand']
    customer_locations = pd.DataFrame(columns=columns)
    capacity = random.randint(capacity_min, capacity_max)

    # Generate random X and Y coordinates for each customer
    for row in range(num_customers):
        x = random.uniform(x_range[0], x_range[1])
        y = random.uniform(y_range[0], y_range[1])
        demand = random.randint(demand_min, demand_max)
        
        # Add the data as new row to the DataFrame
        customer_locations.loc[row] = [x, y, capacity, demand]

    # Insert the depot in the first row
    x_depot = random.uniform(x_range[0], x_range[1])
    y_depot = random.uniform(y_range[0], y_range[1])
    depot = pd.DataFrame({'X': [x_depot], 'Y': [y_depot], 'Vehicle Capacity': capacity, 'Demand': 0})
    customer_locations = pd.concat([depot, customer_locations]).reset_index(drop=True)
    
    # Add total deamand
    customer_locations['Total Demand'] = customer_locations['Demand'].sum()

    return customer_locations

# Function to calculate the distance between two coordinates
def distance(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

# Function to turn seconds into minutes, hours, days
def fun_convert_time(start=None, end=None, seconds=None):
    if(seconds is None): seconds = int(end - start)
    if seconds < 60: # Less than a minute
        computation_time = f'{int(seconds)}s'
    elif seconds < 3600: # Less than an hour (60 * 60 = 3600)
        minutes = seconds // 60
        remaining_seconds = (seconds % 60)
        computation_time = f'{int(minutes)}m, {int(remaining_seconds)}s'
    elif seconds < 86400: # Less than a day (60 * 60 * 24 = 86400)
        hours = seconds // 3600
        remaining_minutes = (seconds % 3600) // 60
        computation_time = f'{int(hours)}h, {int(remaining_minutes)}m'
    else: # More than a day
        days = seconds // 86400
        remaining_hours = (seconds % 86400) // 3600
        computation_time = f'{int(days)}d, {int(remaining_hours)}h'
    return computation_time

# Function to save data as excel sheet
def fun_save_file(data, subfolder_path, name):

    # Select current working directory and subfolder to load the files
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, subfolder_path, name)

    # Save the file
    data.to_excel(file_path)
    
    return print('File saved successfully!')

# Function to read in data
def fun_load_file(subfolder_path, name):

    # Select current working directory and subfolder to load the files
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, subfolder_path, name)

    # Load the file
    return pd.read_excel(io=file_path)



###############################################################################
# Functions for solving TSP/CVRP & Shapley Value
###############################################################################
# Function to compute Shapley value
def fun_shapley_value(player_index, characteristic_function, prints=False):
    """
    Parameters
    ----------
    player_index : int
        index of customer
    characteristic_function : dictionary
        characteristic function of the instance -> all possible subsets of the customers as keys and the respective total costs of the subsets as values
    prints : boolean
        prints interim results for understanding/debugging
    
    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    fun_shapley_value : double
        shapley value of the player

    """
    if (prints == True): 
        if (player_index == 1): print('\n############### SHAPLEY VALUE ###############')
        print('  - Customer: ' + str(player_index))

    # List with highest customer index in all subsets -> take the maximum to get the highest customer index overall
    n = max(max(coalition) for coalition in characteristic_function.keys())
    
    # Check wheater customer/player index is valid
    if (player_index < 1) or (player_index > n):
        raise ValueError('Player index is out of bounds')
    
    # List with all customer/player indices
    players = list(range(1, n + 1))
    shapley_value = 0.0

    # Iterate over all possible coalitions/subsets
    for coalition_size in range(1, n + 1):
        for coalition in itertools.combinations(players, coalition_size):

            # Check wheater player is in subset/coalition
            if player_index in coalition:
                # Get total costs of subset with subset as key; if key is not in the dictionary return 0
                coalition_value = characteristic_function.get(coalition, 0)
                if (prints == True): print('      Subset: {}, Total costs: {}'.format(coalition, coalition_value))

                # Get the total costs of the subset without the player customer/player
                coalition_without_i = set(coalition) - {player_index}
                prev_coalition_value = characteristic_function.get(tuple(sorted(coalition_without_i)), 0)
                if (prints == True): print('        Subset without customer: {}, Total costs: {}'.format(coalition_without_i, prev_coalition_value))

                # Compute marginal costs of customer/player in the subset
                marginal_contribution = coalition_value - prev_coalition_value
                if (prints == True): print('        Marginal contribution: {} - {} = {}'.format(coalition_value, prev_coalition_value, marginal_contribution))
                
                # Shapley value formula
                num_possible_orders = (math.factorial(coalition_size - 1) * math.factorial(n - coalition_size))
                shapley_value += marginal_contribution * (num_possible_orders / math.factorial(n))

    if (prints == True): print('   Shapley value: {}\n'.format(shapley_value))
    
    return shapley_value

# Function to solve TSP with Gurobi
def solve_tsp(coordinates):
    # Create a Gurobi model
    model = gp.Model('TSP')

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
    model.Params.MIPGap = 0.0001 # Stop after Gap smaller than 0.01 %
    model.optimize()

    # Extract the solution
    if model.status == gp.GRB.OPTIMAL:
        solution = [(i, j) for i in range(num_coordinates) for j in range(num_coordinates) if i != j and x[i, j].x > 0.5]
        total_costs = model.objVal
        return solution, total_costs
    else:
        return None, None

# Function to solve CVRP with Gurobi
def solve_cvrp(coordinates, demands, capacity):
    num_nodes = len(coordinates)

    # Create a Gurobi model
    model = gp.Model('CVRP')

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
                #print('Route starts at customer', i)
                sequence = [0,i]
                
                # Search for the next customer, always save the sequence, and ensure it's different from the previous one
                while sequence[-1] != 0:
                    next_customer = [j for j in range(num_nodes) if j != sequence[-1] and x[sequence[-1], j].x > 0.5][0]
                    sequence.append(next_customer)
                
                routes.append(sequence)
                
        # Convert sequence to the same format as it is in the TSP
        routes = [(route[trip], route[trip+1]) for route in routes for trip in range(len(route)-1)]
        
        total_cost = model.objVal
        return routes, total_cost
    else:
        return None, None

# Function to visualize an instance and its optimal solution; additionally you can view cluster assignments of the corresponding model (DBSCAN)
def plot_instance(coord, sequence, total_costs, x_range, y_range, assignments=None, core_point_indices=None, plot_sequence=True, print_sequence=False):
    
    if (print_sequence == True): print('Total costs: {}\nOptimal solution: {}'.format(total_costs, sequence))

    # Depot_coord: tuple with depot X and Y coordinate
    depot_coord = coord[0]

    # Customer_coord: list of tuples with all customers X and Y coordinates; x_coord: list with X coordinates of customers; y_coord: list with Y coordinates of customers
    customer_coord = coord[1:]
    x_coord = np.array([i[0] for i in np.array(customer_coord)])
    y_coord = np.array([i[1] for i in np.array(customer_coord)])

    # Add optimal route with arrows between origin and destination in sequence
    if (plot_sequence == True) & (sequence is not None):
        for trip in sequence:
            origin = trip[0]
            destination = trip[1]
            x = coord[origin][0]
            y = coord[origin][1]
            dx = coord[destination][0] - coord[origin][0]
            dy = coord[destination][1] - coord[origin][1]
            plt.arrow(x=x, y=y, dx=dx, dy=dy, head_width=2, head_length=3, 
                    fc='silver', ec='silver', length_includes_head=True)

    # Create scatter plot with depot (black) and customers (blue)
    plt.scatter(x=depot_coord[0], y=depot_coord[1], color='black', label='Depot', marker='s', s=50)
    if (assignments is None):
        plt.scatter(x=x_coord, y=y_coord, color='blue', label='Customers', marker='o', s=50, zorder=3)

    # Plot customers according to their cluster assignments if parameter 'assignments' is defined
    if (assignments is not None):

        # Get number of clusters and plot customers according to their cluster assignments
        K = len(np.unique(assignments))
        mglearn.discrete_scatter(x1=x_coord, x2=y_coord, y=assignments, markers='o')
        cluster_labels = ['Cluster ' + str(int(i)) for i in np.unique(assignments)]

        # DBSCAN: Mark core points in the plot if the parameter 'core_point_indices' is defined
        if (core_point_indices is not None):

            # Get indices of core points
            core_points_mask = np.zeros_like(assignments, dtype=bool) # Create list with same length as assignments containing only zero/False values
            core_points_mask[core_point_indices] = True # Set core point indices to one/True

            # Mark core points with a dot and add a label as clarification in the legend
            mglearn.discrete_scatter(x1=x_coord[core_points_mask], x2=y_coord[core_points_mask], y=assignments[core_points_mask], markers='.', s=5, c='k')
            last_legend_label = ['Core point']

        else: last_legend_label = [None]
    else: 
        cluster_labels = ['Customers']
        last_legend_label = [None]

    # Add annotations to identify the customers
    for i in range(len(customer_coord)):
        plt.annotate(text='C' + str(i+1), xy=(x_coord[i], y_coord[i]), textcoords='offset points', xytext=(0, 5), ha='center')

    plt.title('Traveling Salesman Problem', size=16)
    plt.xlabel('X', fontweight='bold')
    plt.ylabel('Y', fontweight='bold')
    plt.xticks(range(0, x_range[1] + 10, int(x_range[1]/10))) # Adjust x ticks dynamically to the given x_range
    plt.yticks(range(0, y_range[1] + 10, int(x_range[1]/10))) # Adjust y ticks dynamically to the given y_range
    plt.grid(True, zorder=0)
    
    # Get unique labels and set legend labels dynamically
    legend_labels = ['Depot'] + cluster_labels + last_legend_label
    
    # Get handles and labels from all scatter plots and create the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, legend_labels, loc='best', bbox_to_anchor=(1.05, 1.0))
    plt.show()



###############################################################################
# Functions for cluster features
###############################################################################
# Function to apply the DBSCAN algorithm multiple times on an instance
def fun_multi_dbscan(X, num_customers, model=DBSCAN, prints=False):

    # Define hyperparameters for all instance sizes (number of customers):
    # Tuples with values for 'eps' (max. distance) and 'min_samples' parameter (min number of instances in distance 'eps') to define core points
    if (num_customers <= 7): hyperparameters = [(33, 3), (40, 2), (55, 2)] # 5/6/7 customers
    elif (num_customers <= 10): hyperparameters = [(30, 3), (37, 2), (52, 2)] # 8/9/10 customers
    elif (num_customers <= 12): hyperparameters = [(27, 3), (35, 2), (48, 2)] # 11/12 customers
    else: hyperparameters = [(24, 3), (32, 2), (45, 2)] # 13/14/15 customers

    # Iterate over the hyperparameters tupels
    core_point_indices = []
    for i, tuple in enumerate(hyperparameters):
        # Train DBSCAN with the tuple as parameters 'eps' and 'min_samples'
        dbscan = model(eps=tuple[0], min_samples=tuple[1])

        # First iteration: Train the model on the whole instance to find dense clusters
        if (i == 0):
            assignments = dbscan.fit_predict(X)
            core_points = dbscan.core_sample_indices_ + 1 # Save the core points for the current iteration

        else:
            # Second and third iteration: Train the model on the remaining customers to find less dense clusters
            new_assignments = dbscan.fit_predict(X_residual)

            # Adapt the new cluster labels to the previous cluster labels. E.g. start cluster labelling at three if two previous clusters already existed
            new_assignments[new_assignments != -1] += max(assignments) + 1

            # Update the previous assignments of the outliers when a less dense cluster was found for them
            residual_indices = np.where(np.array(assignments) == -1)[0]
            assignments[assignments == -1] = new_assignments

            # Save the core points for the current iteration
            core_points = np.array(residual_indices)[dbscan.core_sample_indices_] + 1 
        
        # Update core point indices
        core_point_indices += [i for i in core_points - 1]

        # Select only the outliers to try to cluster them in the next iteration
        X_residual = np.array(X)[assignments == -1]
        
        if (prints == True):
            if (i == 0): print('\n############### CLUSTER FEATURES ###############')
            print('- Min cluster size: {} with {} eps\n  labels: {}\n  Core points: {}'.format(tuple[1], tuple[0], assignments, core_points))

        # Stop algorithm if all customers are assigned to a cluster
        if (len(X_residual) == 0):
            if (prints == True): print('-> No outliers left!')
            break
    
    return assignments, core_point_indices

# Function to compute cluster features and add them to the instance
def fun_cluster_features(data, assignments, core_point_indices, features, prints):

    ###############################################################################
    # FEATURE 1, 2 & 3: CLUSTER ID, CORE POINT AND OUTLIER
    # Add cluster labels to the original data
    data['Cluster'] = [np.nan] + list(assignments)

    # Add True/False values depending on wheater a customer is a core point or an outlier
    core_points = np.zeros_like(assignments)
    core_points[core_point_indices] = True
    data['Core Point'] = [np.nan] + list(core_points)
    data['Outlier'] = [np.nan] + list(assignments == -1)

    ###############################################################################
    # FEATURE 4 & 5: NUMBER OF CLUSTERS AND OUTLIERS
    # Create new columns: 'Number Clusters' (exclude outliers) and 'Number Outliers' (count of the first label -1)
    labels = np.unique(assignments)
    data['Number Clusters'] = len(labels[labels != -1])
    data['Number Outliers'] = len(assignments[assignments == -1])

    # Compute centroids of all clusters (exclude the outliers) and get the depot coordinates
    centroids = {cluster: np.array([data[data['Cluster'] == cluster][i].mean() for i in ['X', 'Y']]) for cluster in np.unique(labels[labels != -1])}
    depot_coord = data.loc[0, ['X', 'Y']]

    # Iterate over each cluster (exclude the outliers)
    for cluster_id in labels[labels != -1]:

        # Filter data points belonging to the current cluster
        cluster_data = data[data['Cluster'] == cluster_id]
        
        # Set column 'Core Point' to zero if the cluster consists only out of two customers
        if (len(cluster_data) == 2): data.loc[data['Cluster'] == cluster_id, 'Core Point'] = np.array([0, 0])

        ###############################################################################
        # FEATURE 6: CLUSTER SIZE
        # Assign the cluster size (number of customers) to the corresponding rows in the DataFrame
        data.loc[data['Cluster'] == cluster_id, 'Cluster Size'] = len(cluster_data)

        ###############################################################################
        # FEATURE 7 & 8: CLUSTER CENTROID X AND Y COORDINATE
        # Compute the X and Y coordinates of the cluster centroid
        centroid = centroids[cluster_id]
        data.loc[data['Cluster'] == cluster_id, 'X Centroid'] = centroid[0]
        data.loc[data['Cluster'] == cluster_id, 'Y Centroid'] = centroid[1]

        ###############################################################################
        # FEATURE 9: DISTANCE FROM CUSTOMER TO CENTROID
        # Calculate the distances from each customer to its cluster center (centroid) and assign the calculated distances to the corresponding rows in the DataFrame
        distances = np.linalg.norm(cluster_data[['X', 'Y']] - centroid, axis=1)
        
        data.loc[data['Cluster'] == cluster_id, 'Centroid Distance'] = distances

        if (prints == True):
            print('  - Cluster: {}\n      Centroid: {}\n      Distances to centroid: {}'.format(cluster_id, 
                                                                                                {'X': centroid[0], 'Y': centroid[1]}, 
                                                                                                {'Customer ' + str(customer): distances[i] for i, customer in enumerate(list(data[data['Cluster'] == cluster_id].index))}))

        ###############################################################################
        # FEATURE 10: DISTANCE FROM CENTROID TO DEPOT
        # Calculate the distance from the cluster center to the depot and assign the calculated distances to the corresponding rows in the DataFrame
        distance_to_depot = np.linalg.norm(centroid - depot_coord)
        data.loc[data['Cluster'] == cluster_id, 'Centroid Distance To Depot'] = distance_to_depot

        ###############################################################################
        # FEATURE 11: MINIMUM DISTANCE TO CUSTOMERS OF OTHER CLUSTERS FOR ALL CUSTOMERS IN CURRENT CLUSTER
        # Calculate distances to all customers in other clusters (excluding outliers and depot)
        other_clusters_data = data[(data['Cluster'] != cluster_id) & (data['Cluster'] != -1)].iloc[1:]
        
        # Calculate the distances to the customers of all other clusters if there are at least two clusters in total
        if (len(labels[labels != -1]) >= 2):
            distances_to_other_clusters = cdist(cluster_data[['X', 'Y']], other_clusters_data[['X', 'Y']], metric='euclidean')
            
            # Get the minimum distance for each customer and assign them to the corresponding rows in the DataFrame
            min_distances_to_other_clusters = np.min(distances_to_other_clusters, axis=1)

        # Set the distances to infinity if there is no other cluster
        else: min_distances_to_other_clusters = 100000 # Or np.inf

        data.loc[data['Cluster'] == cluster_id, 'Distance To Closest Other Cluster'] = min_distances_to_other_clusters

        ###############################################################################
        # FEATURE 12: MINIMUM DISTANCE TO CENTROIDS OF OTHER CLUSTERS FOR ALL CUSTOMERS IN CURRENT CLUSTER
        # Calculate distances to all centroids of other clusters (excluding outliers)
        other_cluster_ids = list(set(centroids.keys()) - set([cluster_id]))
        
        # Calculate the distance to all other centroids if there are at least three clusters in total
        if (len(labels[labels != -1]) >= 3):
            distances_to_other_centroids = cdist(cluster_data[['X', 'Y']], [centroids[id] for id in other_cluster_ids], metric='euclidean')
            min_distances_to_other_centroids = np.min(distances_to_other_centroids, axis=1)

        # Calculate the distance to the other centroid if there are only two clusters in total
        elif (len(labels[labels != -1]) == 2):
            min_distances_to_other_centroids = np.linalg.norm(cluster_data[['X', 'Y']] - [centroids[id] for id in other_cluster_ids][0], axis=1)

        # Set the distances to infinity if there is no other cluster
        else: min_distances_to_other_centroids = 100000 # Or np.inf
        
        data.loc[data['Cluster'] == cluster_id, 'Distance To Closest Other Centroid'] = min_distances_to_other_centroids

        ###############################################################################
        # FEATURE 13 & 14: CLUSTER AREA AND DENSITY
        # Calculate the area of the cluster assuming it as circular and then the density
        min_x, min_y = np.min(cluster_data[['X', 'Y']], axis=0)
        max_x, max_y = np.max(cluster_data[['X', 'Y']], axis=0)
        diameter = max(max_x - min_x, max_y - min_y)

        # Apply formulas and assign the calculated values to the corresponding rows in the DataFrame
        area = np.pi * (diameter / 2) ** 2 # Formula for the area of a circle
        density = len(cluster_data) / area # Formula for the density
        data.loc[data['Cluster'] == cluster_id, 'Cluster Area'] = area
        data.loc[data['Cluster'] == cluster_id, 'Cluster Density'] = density
    
    ###############################################################################
    # OUTLIERS: Treat each outlier like a single cluster if there is at least one outlier
    if (len(labels[labels == -1]) >= 1):
        data.loc[data['Cluster'] == -1, ['Centroid Distance', 'Cluster Area']] = 0 # Or np.nan
        data.loc[data['Cluster'] == -1, ['Cluster Size', 'Cluster Area', 'Cluster Density']] = 1 # Or np.nan
        data.loc[data['Cluster'] == -1, 'X Centroid'] = data.loc[data['Cluster'] == -1, 'X']
        data.loc[data['Cluster'] == -1, 'Y Centroid'] = data.loc[data['Cluster'] == -1, 'Y']
        data.loc[data['Cluster'] == -1, 'Centroid Distance To Depot'] = data.loc[data['Cluster'] == -1, 'Depot Distance']
        
        # Get the data of all outliers and all customers in a cluster (excluding depot)
        outlier_data = data.loc[data['Cluster'] == -1]
        all_clusters_data = data[data['Cluster'] != -1].iloc[1:]

        # Compute the distances to all other customers in a cluster for each outlier and get the minimum distance
        if (len(outlier_data) > 1):
            distances_to_other_clusters = cdist(outlier_data[['X', 'Y']], all_clusters_data[['X', 'Y']], metric='euclidean')
            min_distances_to_other_clusters = [min(array) for array in distances_to_other_clusters]
        else:
            distances_to_other_clusters = np.array([np.linalg.norm(outlier_data[['X', 'Y']] - all_clusters_data.iloc[customer][['X', 'Y']]) for customer in range(len(all_clusters_data))])
            min_distances_to_other_clusters = min(distances_to_other_clusters)
        
        # Compute the distances to all cluster centroids for each outlier and get the minimum distance
        distances_to_other_centroids = np.array([[np.linalg.norm(outlier_data.iloc[outlier][['X', 'Y']] - centroid) for centroid in centroids.values()] for outlier in range(len(outlier_data))])
        min_distances_to_other_centroids = [min(array) for array in distances_to_other_centroids]

        # Assign these values to the corresponding rows
        data.loc[data['Cluster'] == -1, 'Distance To Closest Other Cluster'] = min_distances_to_other_clusters
        data.loc[data['Cluster'] == -1, 'Distance To Closest Other Centroid'] = min_distances_to_other_centroids
    
    # Add feature 'Cluster Demand' for CVRP
    if ('Demand' in data.columns): data['Cluster Demand'] = data.groupby('Cluster')['Demand'].transform('sum')

    # Turn columns into integers
    data[['Cluster', 'Core Point', 'Outlier', 'Cluster Size']] = data[['Cluster', 'Core Point', 'Outlier', 'Cluster Size']].apply(pd.to_numeric, errors='coerce').astype('Int64')

    if (prints == True): display(data[features])

    return data