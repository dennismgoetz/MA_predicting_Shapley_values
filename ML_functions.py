import os
import json
import pickle
import time
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib_inline
from IPython.display import display, Markdown
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_validate, GroupKFold, ParameterGrid
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error

###############################################################################
# MAIN SCRIPT FUNCTIONS
###############################################################################
# Function to run a notebook from a parent notebook and update the output cells of it
def run_notebook(notebook_path):
    try:
        # Start the time count and run the notebook given as input
        start = time.time()
        subprocess.run(["jupyter", "nbconvert", "--to", "notebook", 
                        "--execute", "--inplace", notebook_path], check=True)
        
        # Stop the timer make a print indicating the script completion and its run time 
        run_time = fun_convert_time(start=start, end=time.time())
        print(f"Notebook {notebook_path:<19} completed!  Run time: {run_time}")
    
    # Raise a message if an error occurs during the execution of the notebook
    except subprocess.CalledProcessError:
        print(f"Error: Failed to execute the notebook {notebook_path}.")

# Function to define the optimization problem based on the execution context (direct execution or execution by another script)
def fun_load_settings(default_optimization_problem, prints=True):
    try:
        with open("settings.json", "r") as file: # "r" for reading; "rb" for reading binary
            settings = json.load(file)
    except:
        with open("..\settings.json", "r") as file: # Go one folder backwards for the a_tuning_summary.py script
            settings = json.load(file)

    # If the notebook is run by the parent script "main.ipynb", load optimization_problem that is defined in "settings.json"
    if (settings["main_script_execution"] == True):
        optimization_problem = settings["optimization_problem"]
        if prints: print("The notebook was executed by another notebook. :)")

    # If the notebook is run directly, use the manually defined optimization problem
    else:
        optimization_problem = default_optimization_problem
        if prints: print("The notebook is executed directly. :)")
    
    if prints: print(f"Optimization Problem: '{optimization_problem}'")
    return optimization_problem



###############################################################################
# DATASET FUNCTIONS
###############################################################################
# Function to read in data
def fun_load_data(optimization_problem):

    # Start time count for the whole script
    start_script = time.time()

    # Get the name of the folder path and the file name depending on the problem
    if (optimization_problem == "TSP"):
        subfolder_path = "..\\01_data\\01_TSP"
        file_name = "tsp_instances_j_updated.xlsx"
    elif (optimization_problem == "CVRP"):
        subfolder_path = "..\\01_data\\02_CVRP"
        file_name = "cvrp_instances_j_updated.xlsx"
    elif (optimization_problem == "BPP"):
        subfolder_path = "..\\..\\01_data\\03_BPP"
        file_name = "bpp_instances_j_updated.xlsx"
    elif (optimization_problem == "TSP_blended_proxy"): # To create features (Φ DEPOT, Φ MOAT) for blended proxy Φ BLEND
        subfolder_path = "..\\..\\01_data\\01_TSP" # Go one folder further back, as the benchmark scripts are in a subfolder
        file_name = "tsp_instances_j_updated.xlsx"
    elif (optimization_problem == "TSP_benchmarks"): # For evaluation of SHAPO and blended proxy 
        subfolder_path = "..\\..\\01_data\\01_TSP"
        file_name = "tsp_instances_benchmarks.xlsx" # Data set includes SHAPO and Φ BLEND

    # Select current working directory and subfolder to load the file
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, subfolder_path, file_name)

    # Load the file
    return pd.read_excel(io=file_path), start_script

# Function to save data as excel sheet
def fun_save_file(data, subfolder_path, name):
    # Select current working directory and subfolder to load the files
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, subfolder_path, name)

    # Save the file
    data.to_excel(file_path)

# Select columns for training and make train test split
def fun_preprocessing(data, train_size, keep_SHAPO=False):

    ############################## Select features ##############################

    # Remove all features on which a ratio feature is based (e.g. remove "Depot Distance" and keep "Depot Distance Ratio")
    columns = [i for i in data.columns]
    ratio_features = [i for i in columns if "Ratio" in i] # Get all ratio features
    all_other_features = [i for i in columns if i not in ratio_features] # Remove only the ratio features
    columns = [i if (i + " Ratio" not in ratio_features) else i + " Ratio" for i in all_other_features] # Replace the basic features with the ratio features

    # Remove the extracted features
    extracted_features_tsp = ["Unnamed: 0.1", "Unnamed: 0", "Shapley Value Cluster", "Percentage_error", "Percentage Error",
                              "Outlier", "Core Point", "Number Outliers", "Centroid Distance Ratio", "Distance To Closest Other Centroid Ratio", 
                              "X Mean", "Y Mean", "9th CCD Ratio", "10th CCD Ratio"]
    
    # Remove the predictions of the SHAPO approximation by default
    if (keep_SHAPO == False): extracted_features_tsp += ["SHAPO"]

    extracted_features_bin_packing = ["Instance_id_of_instance_size", "Item ID", "0% Percentile Weight", "100% Percentile Weight", "0% Percentile Size", "100% Percentile Size",
                                      "Weight / 0% Percentile Ratio", "Weight / 100% Percentile Ratio", "Size / 0% Percentile Ratio", "Size / 100% Percentile Ratio",
                                      "Weight Mean", "Size Mean", 
                                      "Item Bin Utilization Weight Ratio", "Item Bin Utilization Size Ratio", "Weight Bin Combinations Ratio", "Size Bin Combinations Ratio",
                                      "Weight Quantile Values Ratio", "Size Quantile Values Ratio", "25% Percentile Weight", "50% Percentile Weight", "75% Percentile Weight",
                                      "25% Percentile Size", "50% Percentile Size", "75% Percentile Size", "Weight / 25% Percentile Ratio", "Weight / 50% Percentile Ratio",
                                      "Weight / 75% Percentile Ratio", "Size / 25% Percentile Ratio", "Size / 50% Percentile Ratio", "Size / 75% Percentile Ratio",
                                      "Perfect Weight Bin Combinations Ratio", "Perfect Size Bin Combinations Ratio", "Final Bin Utilization Weight", "Final Bin Utilization Size"]
    extracted_features = extracted_features_tsp + extracted_features_bin_packing
    columns = [i for i in columns if i not in extracted_features]
    train_data = data[columns]

    ############################## Train Test Split ##############################

    # Ensure that the first 700 of the 1000 instances per instance size get into the train set and the remaining 300 instances go into the test set
    entities = "Customers" if ("Number Customers" in train_data.columns) else "Items" # Feature name in TSP and CVRP: "Number Customers", Bin_Packing: "Number Items"
    for size, group in train_data.groupby("Number " + entities): # Do not split instances into train and test set: a instance belongs either to the train or test set
        
        rows = len(group) # For size (Number Customers) = 6: 6000 rows
        number_instances = rows / size # Number of instances: 6000 / 6 = 1000 instances
        number_instances_train = number_instances * train_size # Number of instances in the train set: 1000 * 0.8 = 800 instances for training
        rows_train = int(number_instances_train * size) # Number of rows to select for the train set: 800 * 6 = 4800 rows -> 
        train_group = group.iloc[:rows_train] # Select the first 4800 rows for training
        test_group = group.iloc[rows_train:] # And the remaining 1200 rows for the test set

        # Split X and y for the train and test sets
        features = [i for i in train_data.columns if i != "Shapley Value"]
        X_train_group, X_test_group = train_group[features], test_group[features]
        y_train_group, y_test_group = train_group["Shapley Value"], test_group["Shapley Value"]

        # Define the objects of the first group (Number Customers = 6) as the complete X_train / X_test Data Frames and y_train / y_test Series
        if (size == min(train_data["Number " + entities])):
            X_train, X_test, y_train, y_test = X_train_group, X_test_group, y_train_group, y_test_group
        # Then concat the DataFrames and Series of the following groups to the already existing ones
        else:
            X_train = pd.concat([X_train, X_train_group])
            X_test = pd.concat([X_test, X_test_group])
            y_train = pd.concat([y_train, y_train_group])
            y_test = pd.concat([y_test, y_test_group])

    return X_train, X_test, y_train, y_test, train_data   

# Fit a grid search model, measure the fit time and save best parameter combination into a file
def fun_fit_tuning(search_method, X_train, y_train, file_name):
    # Fit the model on the train data and measure the time
    start = time.time()
    search_method.fit(X_train, y_train)
    total_tuning_time = fun_convert_time(start=start, end=time.time()) # Time for the whole grid search or random grid search process

    # Get param grid or param distribution and the number of all combinations or number of iterations
    if hasattr(search_method, "param_grid"): 
        param_grid = search_method.param_grid
        parameter_combinations = len(ParameterGrid(param_grid))
    elif hasattr(search_method, "param_distributions"): 
        param_grid = search_method.param_distributions
        parameter_combinations = search_method.n_iter

    # Get the type of the search method, the best parameter combination and time counts
    search_type = search_method.__class__.__name__ # Either "GridSearchCV" or "RandomizedSearchCV"
    best_params = search_method.best_params_
    fit_time = fun_convert_time(seconds=search_method.cv_results_["mean_fit_time"].sum()) # Time to fit on each fold during cv, summed over all parameter combinations
    prediction_time = fun_convert_time(seconds=search_method.cv_results_["mean_score_time"].sum()) # Time to predict and evaluate on the test data for each fold, summed over all parameter combinations
    tuning_details = {"Search type": search_type, "Parameter combinations": parameter_combinations, "Total tuning time": total_tuning_time, 
                      "Total tuning fit time": fit_time, "Total tuning prediction time": prediction_time}
    display(tuning_details)

    # Select subfolder and combine the subfolder with the file name to get the folder path
    subfolder = "03_tuning_results/01_files"
    os.makedirs(subfolder, exist_ok=True) # Create subfolder if it doesn't exist

    # Save the parameter grid/distribution
    file_path = os.path.join(subfolder, file_name + "_param_grid.pkl")
    with open(file_path, "wb") as file: # "w" for writring binary
        pickle.dump(param_grid, file)

    # Save the best parameters to a file
    file_path = os.path.join(subfolder, file_name + "_best_params.pkl")
    with open(file_path, "wb") as file:
        pickle.dump(best_params, file)
    
    # Save the run times to a file
    file_path = os.path.join(subfolder, file_name + "_tuning_details.json")
    with open(file_path, "w") as file:
        json.dump(tuning_details, file)

    return tuning_details

# Function to load the best parameter set for a model
def fun_load_best_params(optimization_problem, model_abbreviation):

    # Select subfolder and combine the subfolder with the file name to get the file path
    subfolder = "03_tuning_results/01_files"
    file_name = optimization_problem + "_" + model_abbreviation + "_best_params.pkl"
    file_path = os.path.join(subfolder, file_name)

    # Load the file and show the best parameters
    with open(file_path, "rb") as file:
        best_params = pickle.load(file)
    display(best_params)

    return best_params



###############################################################################
# TIME FUNCTIONS
###############################################################################
# Function to turn seconds into minutes, hours, days
def fun_convert_time(start=None, end=None, seconds=None):
    if(seconds is None): seconds = int(end - start)
    if seconds < 60: # Less than a minute
        computation_time = f"{int(seconds)}s"
    elif seconds < 3600: # Less than an hour (60 * 60 = 3600)
        minutes = seconds // 60
        remaining_seconds = (seconds % 60)
        computation_time = f"{int(minutes)}m, {int(remaining_seconds)}s"
    elif seconds < 86400: # Less than a day (60 * 60 * 24 = 86400)
        hours = seconds // 3600
        remaining_minutes = (seconds % 3600) // 60
        computation_time = f"{int(hours)}h, {int(remaining_minutes)}m"
    else: # More than a day
        days = seconds // 86400
        remaining_hours = (seconds % 86400) // 3600
        computation_time = f"{int(days)}d, {int(remaining_hours)}h"
    return computation_time



###############################################################################
# SCALING FUNCTIONS
###############################################################################
# Function to compute the scaled MAPE  in a CV process
# (Scale the predictions, such that the sum of all predicitons per instance is equal to the sum of the Shapley values of that instance)
def fun_scaled_neg_MAPE(estimator, X, y_true):
    # Make predictions
    y_pred = estimator.predict(X)

    # Connect the X_predict Data Frame with the true y labels of y_true; then assighn the predictions as a columns to the Data Frame
    Xy_train = pd.merge(left=X, right=y_true, left_index=True, right_index=True)
    Xy_train_pred = Xy_train.assign(Predictions=pd.Series(data=y_pred, index=X.index))

    # Compute the sum of predicted Shapley values and the MAPE without scaling
    Xy_train_pred["Sum of Predictions"] = Xy_train_pred.groupby("Instance ID")["Predictions"].transform("sum")
    Xy_train_pred["MAPE (original)"] = np.abs((y_true - y_pred) / y_true)
    
    # Scale the predictions with the sum of the Shapley Values and receive the new predictions (the sum of the predicted Shapley values should be equal to the total costs of an instance)
    unit = "Costs" if ("Total Costs" in X.columns) else "Bins" # Feature name in TSP and CVRP: "Total Costs", Bin_Packing: "Total Bins"
    entities = "Customers" if ("Number Customers" in X.columns) else "Items"
    y_pred = Xy_train_pred["Predictions"] * (Xy_train_pred["Total " + unit] / Xy_train_pred["Sum of Predictions"])

    # Put the new predictions in column "Improved Predictions" and compute the new MAPE with scaling; optionally view the Data Frame Xy_train_pred (only possible if n_jobs is not -1 during CV)
    Xy_train_pred["Improved Predictions"] = y_pred
    Xy_train_pred["MAPE (scaled)"] = np.abs((y_true - y_pred) / y_true)
    #display(Xy_train_pred[["Instance ID", "Number " + entities, "Total " + unit, "Sum of Predictions", "Predictions", "Improved Predictions", "Shapley Value", "MAPE (original)", "MAPE (scaled)"]].sort_index().head(12))
    
    return - np.mean(np.abs((y_true - y_pred) / y_true))

# Function to compute the scaled RMSE in a CV process
def fun_scaled_neg_RMSE(estimator, X, y_true):
    # Make predictions
    y_pred = estimator.predict(X)
    
    # Connect the X_predict Data Frame with the true y labels of y_true; then assighn the predictions as a columns to the Data Frame
    Xy_train = pd.merge(left=X, right=y_true, left_index=True, right_index=True)
    Xy_train_pred = Xy_train.assign(Predictions=pd.Series(data=y_pred, index=X.index))
    
    # Compute the sum of predicted Shapley values
    Xy_train_pred["Sum of Predictions"] = Xy_train_pred.groupby("Instance ID")["Predictions"].transform("sum")
    
    # Scale the predictions with the sum of the Shapley Values and receive the new predictions (the sum of the predicted Shapley values should be equal to the total costs of an instance)
    unit = "Costs" if ("Total Costs" in X.columns) else "Bins" # Feature name in TSP and CVRP: "Total Costs", Bin_Packing: "Total Bins"
    y_pred = Xy_train_pred["Predictions"] * (Xy_train_pred["Total " + unit] / Xy_train_pred["Sum of Predictions"])

    return - np.sqrt(np.mean((y_true - y_pred)**2))

# Function make predictions with a model, scale the predictions and compute the MAPE and RMSE for the test set
def fun_predict_with_scaling(model, X_train, y_train, X_test, y_test, apply_scaling):
    
    # Fit model on train data and get predictions for X_test (measure the fit and prediction time)
    try: # If the model is already fitted (e.g. a grid search model after the tuning and refit=True), you can directly make the predictions
        check_is_fitted(model)
        start = time.time()
        y_pred = model.predict(X_test)
        prediction_time = fun_convert_time(start=start, end=time.time())
        fit_time = None
    except NotFittedError:
        start = time.time()
        model.fit(X_train, y_train)
        fit_time = fun_convert_time(start=start, end=time.time())

        start = time.time()
        y_pred = model.predict(X_test)
        prediction_time = fun_convert_time(start=start, end=time.time())
    
    # Improve predictions: Sum of predicted Shapley values must be equal to the total costs for all instances
    if (apply_scaling == True):
        
        # Connect the X_predict Data Frame with the true y labels of y_test; then assighn the predictions as a columns to the Data Frame
        Xy_train = pd.merge(left=X_test, right=y_test, left_index=True, right_index=True)
        Xy_train_pred = Xy_train.assign(Predictions=pd.Series(data=y_pred, index=X_test.index))
        
        # Compute the sum of predicted Shapley values
        Xy_train_pred["Sum of Predictions"] = Xy_train_pred.groupby("Instance ID")["Predictions"].transform("sum")
        
        # Scale the predictions with the sum of the Shapley Values and receive the new predictions (the sum of the predicted Shapley values should be equal to the total costs of an instance)
        unit = "Costs" if ("Total Costs" in X_train.columns) else "Bins" # Feature name in TSP and CVRP: "Total Costs", Bin_Packing: "Total Bins"
        entities = "Customers" if ("Number Customers" in X_train.columns) else "Items"
        y_pred = Xy_train_pred["Predictions"] * (Xy_train_pred["Total " + unit] / Xy_train_pred["Sum of Predictions"])
  
        # Put the new predictions in column "Improved Predictions"; optionally view the Data Frame Xy_train_pred
        Xy_train_pred["Improved Predictions"] = y_pred
        #display(Xy_train_pred[["Instance ID", "Number " + entities, "Total " + unit, "Sum of Predictions", "Predictions", "Improved Predictions", "Shapley Value"]].sort_index().head(12))
    
    # If the scaling is not applied, just add the correct indices to the predictions for the categorical scores later on
    else: y_pred = pd.Series(data=y_pred, index=X_test.index)

    # Compute errors
    MAPE_score = np.round(mean_absolute_percentage_error(y_true=y_test, y_pred=y_pred), 6) * 100
    RMSE_score = np.round(root_mean_squared_error(y_true=y_test, y_pred=y_pred), 4)

    return MAPE_score, RMSE_score, y_pred, fit_time, prediction_time



###############################################################################
# SCORING FUNCTIONS
###############################################################################
# Function to view grid search CV scores of all parameter combinations
def fun_tuning_results(search_method, search_space):
    # Turn results into a data frame and modify the "mean_fit_time" column for better understanding
    results_df = pd.DataFrame(search_method.cv_results_).sort_values(by="mean_test_score", ascending=False).reset_index(drop=True)
    results_df["converted_mean_fit_time"] = results_df["mean_fit_time"].apply(lambda X: fun_convert_time(seconds=X))

    # In case search_space is a dictionary, take all the keys. If it is a list of dictionaries, take the keys of all the dictionaries in the list
    if (isinstance(search_space, dict)): keys = list(search_space.keys())
    else: keys = set(key for dictionary in search_space for key in dictionary.keys())

    # Select all the parameter columns that appear in the grid search as well as the test score and fit time
    columns = ["param_" + i for i in keys] + ["mean_test_score", "converted_mean_fit_time"]
    results_df = results_df[columns]

    # Modify columns names in the Data Frame
    columns = [i.replace("param_", "") for i in columns] # Remove the string "param_" at the beginning: "param_scaler": "scaler"
    columns = [name.split("__")[-1] for name in columns] # Remove the model name before "__": "mlpregressor__alpha" -> "alpha"
    results_df.columns = columns

    # Display multiple Data Frames if the results_df has too many rows (max is 60 rows for displaying all rows)
    display(Markdown("**Cross validation scores of different parameter combinations:**"))
    if len(results_df) > 60:
        for i in range((len(results_df) + 59) // 60): # Ensure to always round up
            start = i * 60
            end = min(start + 59, len(results_df) - 1)  # Avoid that the index gets out of bounds
            display(results_df.loc[start:end])
    else: display(results_df)

# Compute train and test scores
def fun_scores(model, X_train, y_train, X_test=None, y_test=None, apply_scaling=True, compute_test_scores=False):

    # Get CV train scores of a grid search model
    if (hasattr(model, "best_score_")):
        MAPE_train = - np.round(model.best_score_, 6) * 100
        RMSE_train, cv_computation_time = None, None
        print("CV MAPE (scaled) train data:  {} %".format(MAPE_train)) 
        
        # Show best parameter combination
        display(Markdown("**Best model / parameter combination:**"))
        if (len(model.get_params()) <= 10): display(model.best_estimator_)
        else: display(model.best_params_)

    # Compute CV train scores if model is a usual estimator and measure CV computation time
    else:
        # Get MAPE and RMSE scores from the model's scaled predictions and unscaled predictions
        start = time.time()
        cv_scores = cross_validate(estimator=model, X=X_train, y=y_train, n_jobs=-1, 
                                   cv=GroupKFold(n_splits=3).split(X_train, y_train, groups=X_train["Instance ID"]), # Cross-validation with GroupKFold to keep instances together in one fold
                                   scoring={"scaled_mape": fun_scaled_neg_MAPE,
                                            "scaled_rmse": fun_scaled_neg_RMSE,
                                            "original_neg_mape": "neg_mean_absolute_percentage_error",
                                            "original_neg_rmse": "neg_root_mean_squared_error"})
        cv_computation_time = fun_convert_time(start=start, end=time.time())

        # Print train scores for either the scaled predictions or the unscaled predictions
        if (apply_scaling == True):
            MAPE_train = - np.round(cv_scores["test_scaled_mape"].mean(), 6) * 100
            RMSE_train = - np.round(cv_scores["test_scaled_rmse"].mean(), 4)
        else:
            MAPE_train = - np.round(cv_scores["test_original_neg_mape"].mean(), 6) * 100
            RMSE_train = - np.round(cv_scores["test_original_neg_rmse"].mean(), 4)
        print("CV MAPE ({}) train data:  {} %".format("scaled" if apply_scaling else "original", MAPE_train))
        print("CV RMSE ({}) train data: {}".format("scaled" if apply_scaling else "original", RMSE_train))        
        print("CV computation time:", cv_computation_time)
    
    # Compute test scores if compute_test_scores == True
    if (compute_test_scores == True):
        if (X_test is None) or (y_test is None): raise ValueError("You need to define X_test and y_test to compute the test scores.")
        # Get MAPE and RMSE scores from the model's scaled predictions and update scores
        MAPE_test, RMSE_test, y_pred, fit_time, prediction_time = fun_predict_with_scaling(model, X_train, y_train, X_test, y_test, apply_scaling)
        MAPE = {"Train data": MAPE_train, "Test data": MAPE_test}
        RMSE = {"Train data": RMSE_train, "Test data": RMSE_test}

        # Compute error measures for each instance size group individually
        # Group X by instance size and apply for each group the error measure fct. Use indices of each group to select the regarding true y values and the improved predictions
        entities = "Customers" if ("Number Customers" in X_train.columns) else "Items" # Feature name in TSP and CVRP: "Number Customers", Bin_Packing: "Number Items"
        MAPE_cat = X_test.groupby(by="Number " + entities).apply(lambda group: mean_absolute_percentage_error(y_true=y_test.loc[group.index], y_pred=y_pred.loc[group.index]))
        RMSE_cat = X_test.groupby(by="Number " + entities).apply(lambda group: root_mean_squared_error(y_true=y_test.loc[group.index], y_pred=y_pred.loc[group.index]))

        # Round results and merge them into a data frame
        MAPE_cat = np.round(MAPE_cat, 6) * 100
        RMSE_cat = np.round(RMSE_cat, 4)
        df = pd.DataFrame(data=[MAPE_cat, RMSE_cat], index=["MAPE", "RMSE"])
        df["Mean"] = [MAPE_test, RMSE_test]

        # Print results and show data frame of instance size groups
        print("\nMAPE ({}) test data:  {} %".format("scaled" if apply_scaling else "original", MAPE_test))
        print("RMSE ({}) test data: {}".format("scaled" if apply_scaling else "original", RMSE_test))
        if (fit_time is not None): print("Model fit time:", fit_time)
        print("Model prediction time:", prediction_time)
        display(Markdown("**MAPE and RMSE on test data per instance size:**")), display(df)

        return {"MAPE": MAPE, "RMSE": RMSE, "CV computation time": cv_computation_time, 
                "Model fit time": fit_time, "Model prediction time": prediction_time, "Scores per instance size": df}

    else: return {"MAPE": MAPE_train, "RMSE": RMSE_train, "CV computation time": cv_computation_time}



###############################################################################
# FEATURE FUNCTIONS
###############################################################################
# View feature importance of the tree based models
def plot_feature_weights(model, n_features):

    # Get model name
    model_name = type(model).__name__

    # Get feature weights
    if (hasattr(model, "coef_")):
        weights = np.abs(model.coef_)
        x_label = "Absolute Feature Coefficient"
    elif (hasattr(model, "feature_importances_")):
        weights = model.feature_importances_
        x_label = "Feature Importance"
    else: 
        print("Error: Plotting feature weights was not possible.")
        print("       Model has neither attribute '.coef' nor attribute '.feature_importances'.")
        return

    # Get feature names, number of features and create a dictionary with feature names as keys and their weights as values
    feature_names = model.feature_names_in_
    weights_dict = {str(feature): weight for feature, weight in zip(feature_names, weights)}

    # Show only the features with the 20 highest weights
    weights_dict = dict(sorted(weights_dict.items(), key=lambda item: item[1])[-n_features:])
    
    # Set LaTeX style fonts in matplotlib
    plt.rc("font", family="serif", serif="Computer Modern")
    plt.rc("text", usetex=True)
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

    # Visualize weights of features
    plt.figure(figsize=(7.5, n_features/4))
    plt.barh(y=range(n_features), width=weights_dict.values(), align="center", 
             color=plt.cm.viridis(np.linspace(0.2, 0.8, n_features)))

    # Use LaTeX commands for bold text in the title and labels
    plt.xlabel(x_label, size=10, fontweight="bold")
    plt.xlabel(rf"\textbf{{{x_label}}}", size=12)
    plt.ylabel(r"\textbf{Feature}", size=12)
    plt.title(rf"\textbf{{Feature Importances of {model_name}}}", size=16, color="darkblue")

    plt.yticks(np.arange(n_features), weights_dict.keys())
    plt.grid(axis="x", linestyle="--", alpha=0.75)
    plt.show()

    # Set matplotlib style fonts to default again
    plt.rcdefaults()