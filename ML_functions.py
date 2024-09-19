import os
import pickle
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib_inline
from IPython.display import display
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

###############################################################################
# DATASET FUNCTIONS
###############################################################################
# Function to read in data
def fun_load_data(optimization_problem):

    # Get the name of the folder and the file to store the final DataFrame
    if (optimization_problem == 'TSP'):
        subfolder_path = '..\\01_data\\01_TSP'
        file_name = 'tsp_instances_j_updated.xlsx'
    elif (optimization_problem == 'CVRP'):
        subfolder_path = '..\\01_data\\02_CVRP'
        file_name = 'cvrp_instances_j_updated.xlsx'
    elif (optimization_problem == 'Bin_Packing'):
        subfolder_path = '..\\..\\01_data\\03_bin_packing'
        file_name = 'bin_packing_instances_j_updated.xlsx'

    # Select current working directory and subfolder to load the file
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, subfolder_path, file_name)

    # Load the file
    return pd.read_excel(io=file_path)

# Function to save data as excel sheet
def fun_save_file(data, subfolder_path, name):
    # Select current working directory and subfolder to load the files
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, subfolder_path, name)

    # Save the file
    data.to_excel(file_path)

# Select columns for training and split dataset into features and target (shapley value)
def fun_preprocessing(data):

    # Remove all features on which a ratio feature is based (e.g. remove 'Depot Distance' and keep 'Depot Distance Ratio')
    columns = [i for i in data.columns]
    ratio_features = [i for i in columns if 'Ratio' in i] # Get all ratio features
    all_other_features = [i for i in columns if i not in ratio_features] # Remove only the ratio features
    columns = [i if (i + ' Ratio' not in ratio_features) else i + ' Ratio' for i in all_other_features] # Replace the basic features with the ratio features

    # Remove the extracted features
    extracted_features_tsp = ['Unnamed: 0.1', 'Unnamed: 0', 'Shapley Value Cluster', 'SHAPO', 'Percentage_error', 'Percentage Error',
                              'Outlier', 'Core Point', 'Number Outliers', 'Centroid Distance Ratio', 'Distance To Closest Other Centroid Ratio', 
                              'X Mean', 'Y Mean', '9th CCD Ratio', '10th CCD Ratio']
    extracted_features_bin_packing = ['Instance_id', '0% Percentile Weight', '100% Percentile Weight', '0% Percentile Size', '100% Percentile Size',
                                      'Weight / 0% Percentile Ratio', 'Weight / 100% Percentile Ratio', 'Size / 0% Percentile Ratio', 'Size / 100% Percentile Ratio',
                                      'Weight Mean', 'Size Mean',
                                      'Item Bin Utilization Weight Ratio', 'Item Bin Utilization Size Ratio', 'Weight Bin Combinations Ratio', 'Size Bin Combinations Ratio',
                                      'Weight Quantile Values Ratio', 'Size Quantile Values Ratio', '25% Percentile Weight', '50% Percentile Weight', '75% Percentile Weight',
                                      '25% Percentile Size', '50% Percentile Size', '75% Percentile Size', 'Weight / 25% Percentile Ratio', 'Weight / 50% Percentile Ratio',
                                      'Weight / 75% Percentile Ratio', 'Size / 25% Percentile Ratio', 'Size / 50% Percentile Ratio', 'Size / 75% Percentile Ratio']
                                    #'25% Percentile Size Ratio', '50% Percentile Size Ratio', '75% Percentile Size Ratio', '100% Percentile Size Ratio' # These columns were deleted from the original bin packing Data Frame
    extracted_features = extracted_features_tsp + extracted_features_bin_packing
    columns = [i for i in columns if i not in extracted_features]
    
    # Select columns and split X and y
    train_data = data[columns]
    X = train_data[[i for i in train_data.columns if not i == 'Shapley Value']]
    y = train_data['Shapley Value']

    return X, y, train_data    

# Fit a grid search model, measure the fit time and save best parameter combination into a file
def fun_fit_tuning(search_method, X_train, y_train, file_name):
    # Fit the model on the train data and measure the time
    start = time.time()
    search_method.fit(X_train, y_train)
    fit_time = fun_convert_time(start=start, end=time.time())
    print('Tuning fit time:', fit_time)

    # Get param grid or param distribution
    if hasattr(search_method, 'param_grid'): param_grid = search_method.param_grid
    elif hasattr(search_method, 'param_distributions'): param_grid = search_method.param_distributions

    # Get best parameter combination
    best_params = search_method.best_params_

    # Select subfolder and combine the subfolder with the file name to get the folder path
    subfolder = '03_best_parameters'
    os.makedirs(subfolder, exist_ok=True) # Create subfolder if it doesn't exist

    # Save the parameter grid/distribution
    file_path = os.path.join(subfolder, file_name + '_param_grid.pkl')
    with open(file_path, 'wb') as file:
        pickle.dump(param_grid, file)

    # Save the best parameters to a file
    file_path = os.path.join(subfolder, file_name + '_best_params.pkl')
    with open(file_path, 'wb') as file:
        pickle.dump(best_params, file)

    return fit_time

# Function to load the best parameter set for a model
def fun_load_best_params(file_name):

    # Select subfolder and combine the subfolder with the file name to get the folder path
    subfolder = '03_best_parameters'
    file_path = os.path.join(subfolder, file_name)

    # Load the file and show the best parameters
    with open(file_path, 'rb') as file:
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
    
    # Compute the sum of predicted Shapley values and the sum of true Shapley values (the sum of the predicted Shapley values should be equal to the total costs/sum of all Shapley values of an instance)
    Xy_train_pred['Sum of Predictions'] = Xy_train_pred.groupby('Instance ID')['Predictions'].transform('sum')
    Xy_train_pred['Sum of Costs (Shapley values)'] = Xy_train_pred.groupby('Instance ID')['Shapley Value'].transform('sum')
    
    # Compute new predictions
    y_pred = Xy_train_pred['Predictions'] * (Xy_train_pred['Sum of Costs (Shapley values)'] / Xy_train_pred['Sum of Predictions'])

    return - np.mean(np.abs((y_true - y_pred) / y_true))

# Function to compute the scaled RMSE in a CV process
def fun_scaled_neg_RMSE(estimator, X, y_true):
    # Make predictions
    y_pred = estimator.predict(X)
    
    # Connect the X_predict Data Frame with the true y labels of y_true; then assighn the predictions as a columns to the Data Frame
    Xy_train = pd.merge(left=X, right=y_true, left_index=True, right_index=True)
    Xy_train_pred = Xy_train.assign(Predictions=pd.Series(data=y_pred, index=X.index))
    
    # Compute the sum of predicted Shapley values and the sum of true Shapley values (the sum of the predicted Shapley values should be equal to the total costs/sum of all Shapley values of an instance)
    Xy_train_pred['Sum of Predictions'] = Xy_train_pred.groupby('Instance ID')['Predictions'].transform('sum')
    Xy_train_pred['Sum of Costs (Shapley values)'] = Xy_train_pred.groupby('Instance ID')['Shapley Value'].transform('sum')
    
    # Compute new predictions
    y_pred = Xy_train_pred['Predictions'] * (Xy_train_pred['Sum of Costs (Shapley values)'] / Xy_train_pred['Sum of Predictions'])

    return - np.sqrt(np.mean((y_true - y_pred)**2))

# Function make predictions with a model, scale the predictions and compute the MAPE and RMSE for the train or test set
def fun_predict_with_scaling(model, X_train, y_train, X_predict, y_true, apply_scaling):
    
    # Fit model on train data and get predictions for X_predict (X_predict usually is X_test, but the prediction for X_train is also possible to get the train score)
    try: # If the model is already fitted (e.g. a grid search model after the tuning), you can directly make the predictions
        y_pred = model.predict(X_predict)
        fit_time = None
    except:
        start = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_predict)
        fit_time = fun_convert_time(start=start, end=time.time())
    
    # Improve predictions: Sum of predicted Shapley values must be equal to the total costs for all instances
    if (apply_scaling == True):
        
        # Connect the X_predict Data Frame with the true y labels of y_true; then assighn the predictions as a columns to the Data Frame
        Xy_train = pd.merge(left=X_predict, right=y_true, left_index=True, right_index=True)
        Xy_train_pred = Xy_train.assign(Predictions=pd.Series(data=y_pred, index=X_predict.index))
        
        # Compute the sum of predicted Shapley values and the sum of true Shapley values (the sum of the predicted Shapley values should be equal to the total costs/sum of all Shapley values of an instance)
        Xy_train_pred['Sum of Predictions'] = Xy_train_pred.groupby('Instance ID')['Predictions'].transform('sum')
        Xy_train_pred['Sum of Costs (Shapley values)'] = Xy_train_pred.groupby('Instance ID')['Shapley Value'].transform('sum')
        
        # Compute new predictions in column 'Improved Predictions' and get all predictions as a pd.Series; optionally view the Data Frame Xy_train_pred
        Xy_train_pred['Improved Predictions'] = Xy_train_pred['Predictions'] * (Xy_train_pred['Sum of Costs (Shapley values)'] / Xy_train_pred['Sum of Predictions'])
        y_pred = Xy_train_pred['Improved Predictions']
        #display(Xy_train_pred[['Instance ID', 'Number Customers', 'Total Costs', 'Sum of Costs (Shapley values)', 'Predictions', 'Sum of Predictions', 'Improved Predictions', 'Shapley Value']].sort_index().head(12))
    
    # If the scaling is not applied, just add the correct indices to the predictions for the categorical scores later on
    else: y_pred = pd.Series(data=y_pred, index=X_predict.index)

    # Compute errors
    MAPE_score = np.round(mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred), 6) * 100
    RMSE_score = np.round(mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False), 4)

    return MAPE_score, RMSE_score, y_pred, fit_time



###############################################################################
# SCORING FUNCTIONS
###############################################################################
# Function to view grid search CV scores of all parameter combinations
def fun_tuning_results(search_method, search_space):
    # Turn results into a data frame and modify the 'mean_fit_time' column for better understanding
    results_df = pd.DataFrame(search_method.cv_results_).sort_values(by='mean_test_score', ascending=False).reset_index(drop=True)
    results_df['converted_mean_fit_time'] = results_df['mean_fit_time'].apply(lambda X: fun_convert_time(seconds=X))
    
    # Display multiple Data Frames if the results_df has too many rows (max is 60 rows for displaying all rows)
    print('Cross validation scores of different parameter combinations:')
    if len(results_df) > 60:
        for i in range((len(results_df) // 60) + 1):
            start = i * 60
            end = start + 59
            # Select all the parameter columns that appear in the grid search as well as the test score and fit time
            display(results_df[['param_' + i for i in list(search_space.keys())] + ['mean_test_score', 'converted_mean_fit_time']].loc[start:end])
    return results_df

# Compute train and test scores
def fun_scores(model, X_train, y_train, X_test=None, y_test=None, apply_scaling=True, compute_test_scores=False):

    # Get CV train scores of a grid search model
    if (hasattr(model, 'best_score_')):
        MAPE_train = - np.round(model.best_score_, 6) * 100
        RMSE_train, cv_computation_time = None, None
        print('CV MAPE (scaled) train data:  {} %'.format(MAPE_train)) 
        
        # Show best parameter combination
        print('\nBest model / parameter combination:')
        if (len(model.get_params()) <= 10): display(model.best_estimator_)
        else: display(model.best_params_)

    # Compute CV train scores if model is a usual estimator and measure CV computation time
    else:
        # Get MAPE and RMSE scores from the model's scaled predictions and unscaled predictions
        start = time.time()
        cv_scores = cross_validate(estimator=model, X=X_train, y=y_train, cv=3, n_jobs=-1,
                                   scoring={'scaled_mape': fun_scaled_neg_MAPE,
                                            'scaled_rmse': fun_scaled_neg_RMSE,
                                            'original_neg_mape': 'neg_mean_absolute_percentage_error',
                                            'original_neg_rmse': 'neg_root_mean_squared_error'})
        cv_computation_time = fun_convert_time(start=start, end=time.time())

        # Print train scores for either the scaled predictions or the unscaled predictions
        if (apply_scaling == True):
            MAPE_train = - np.round(cv_scores['test_scaled_mape'].mean(), 6) * 100
            RMSE_train = - np.round(cv_scores['test_scaled_rmse'].mean(), 4)
        else:
            MAPE_train = - np.round(cv_scores['test_original_neg_mape'].mean(), 6) * 100
            RMSE_train = - np.round(cv_scores['test_original_neg_rmse'].mean(), 4)
        print('CV MAPE ({}) train data:  {} %'.format('scaled' if apply_scaling else 'original', MAPE_train))
        print('CV RMSE ({}) train data: {}'.format('scaled' if apply_scaling else 'original', RMSE_train))        
        print('CV computation time:', cv_computation_time)
    
    # Compute test scores if compute_test_scores == True
    if (compute_test_scores == True):
        if (X_test is None) or (y_test is None): raise ValueError("You need to define X_test and y_test to compute the test scores.")
        # Get MAPE and RMSE scores from the model's scaled predictions and update scores
        MAPE_test, RMSE_test, y_pred, fit_time = fun_predict_with_scaling(model, X_train, y_train, X_test, y_test, apply_scaling)
        MAPE = {'Train data': MAPE_train, 'Test data': MAPE_test}
        RMSE = {'Train data': RMSE_train, 'Test data': RMSE_test}

        # Compute error measures for each instance size group individually
        # Group X by instance size and apply for each group the error measure fct. Use indices of each group to select the regarding true y values and the improved predictions
        entities = 'Customers' if ('Number Customers' in X_train.columns) else 'Items' # Feature name in TSP and CVRP: 'Number Customers', Bin_Packing: 'Number Items'
        MAPE_cat = X_test.groupby(by='Number ' + entities).apply(lambda group: mean_absolute_percentage_error(y_true=y_test.loc[group.index], y_pred=y_pred.loc[group.index]))
        RMSE_cat = X_test.groupby(by='Number ' + entities).apply(lambda group: mean_squared_error(y_true=y_test.loc[group.index], y_pred=y_pred.loc[group.index], squared=False))

        # Round results and merge them into a data frame
        MAPE_cat = np.round(MAPE_cat, 6) * 100
        RMSE_cat = np.round(RMSE_cat, 4)
        df = pd.DataFrame(data=[MAPE_cat, RMSE_cat], index=['MAPE', 'RMSE'])
        df['Mean'] = [MAPE_test, RMSE_test]

        # Print results and show data frame of instance size groups
        print('\nMAPE ({}) test data:  {} %'.format('scaled' if apply_scaling else 'original', MAPE_test))
        print('RMSE ({}) test data: {}'.format('scaled' if apply_scaling else 'original', RMSE_test))
        if (fit_time is not None): print('Model fit time:', fit_time)
        print('\nMAPE and RMSE on test data per instance size:'), display(df)

        return {'MAPE': MAPE, 'RMSE': RMSE, 'CV computation time': cv_computation_time, 'Fit model time': fit_time, 'Scores per instance size': df}

    else: return {'MAPE': MAPE_train, 'RMSE': RMSE_train, 'CV computation time': cv_computation_time}


###############################################################################
# FEATURE FUNCTIONS
###############################################################################
# View feature importance of the tree based models
def plot_feature_weights(model, n_features):

    # Get feature weights
    if (hasattr(model, 'coef_')):
        weights = np.abs(model.coef_)
        x_label = 'Absolute Feature Coefficients'
    elif (hasattr(model, 'feature_importances_')):
        weights = model.feature_importances_
        x_label = 'Feature Importances'

    # Get feature names, number of features and create a dictionary with feature names as keys and their weights as values
    feature_names = model.feature_names_in_
    weights_dict = {str(feature): weight for feature, weight in zip(feature_names, weights)}

    # Show only the features with the 20 highest weights
    weights_dict = dict(sorted(weights_dict.items(), key=lambda item: item[1])[-n_features:])
    
    # Visualize weights of features
    plt.figure(figsize=(15, n_features/5))
    plt.barh(y=range(n_features), width=weights_dict.values(), align='center', color='forestgreen')
    plt.yticks(np.arange(n_features), weights_dict.keys())
    plt.xlabel(x_label, size=10, fontweight='bold')
    plt.ylabel('Feature', size=10, fontweight='bold')
    plt.title('Feature Importances', size=15, fontweight='bold')
    plt.grid(axis='x')
    plt.show()