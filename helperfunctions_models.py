import os
import pickle
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib_inline
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

###############################################################################
# DATASET FUNCTIONS
###############################################################################
# Function to read in data
def fun_load_file(subfolder_path, name):

    # Select current working directory and subfolder to load the files
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, subfolder_path, name)

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
    # Select columns
    columns = [i for i in data.columns]
    extracted_features = ['Unnamed: 0.1', 'Unnamed: 0', 'Shapley Value Cluster', 'SHAPO', 'Percentage_error', 'Percentage Error']
    train_features = [i for i in columns if i not in extracted_features]
    train_data = data[train_features]

    # Split X and y
    X = train_data[[i for i in train_data.columns if not i == 'Shapley Value']]
    y = train_data['Shapley Value']

    return X, y, train_data    

# # Function to edit a column in the dataset
# def fun_edit_data(data, column, column_name, problem='TSP', file_name='combined_train_instances_dennis'):
#     data[column_name] = column
#     if (problem == 'TSP'):
#         data.to_excel('../01_data/01_TSP/' + str(file_name) + '.xlsx')
#     elif (problem == 'CVRP'):
#         data.to_excel('../01_data/02_CVRP/' + str(file_name) + '.xlsx')

# Fit a grid search model, measure the fit time and save best parameter combination as into a file
def fun_fit_tuning(search_method, X_train, y_train, file_name):
    # Fit the model on the train data and measure the time
    start = time.time()
    search_method.fit(X_train, y_train)
    fit_time = fun_convert_time(start=start, end=time.time())

    # Get best parameter combination
    best_params = search_method.best_params_

    # Select subfolder and combine the subfolder with the file name to get the folder path
    subfolder = '02_best_parameters'
    os.makedirs(subfolder, exist_ok=True) # Create subfolder if it doesn't exist
    file_path = os.path.join(subfolder, file_name)

    # Save the best parameters to a file
    with open(file_path, 'wb') as file:
        pickle.dump(best_params, file)

    return fit_time

# Function to load the best parameter set for a model
def fun_load_best_params(file_name):

    # Select subfolder and combine the subfolder with the file name to get the folder path
    subfolder = '02_best_parameters'
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
# SCORING FUNCTION
###############################################################################
# Function to view grid search CV scores of all parameter combinations
def fun_tuning_results(search_method, search_space):
    # Turn results into a data frame and modify the 'mean_fit_time' column for better understanding
    results_df = pd.DataFrame(search_method.cv_results_).sort_values(by='mean_test_score', ascending=False).reset_index(drop=True)
    results_df['converted_mean_fit_time'] = results_df['mean_fit_time'].apply(lambda X: fun_convert_time(seconds=X))

    # Select all the parameter columns that appear in the grid search as well as the test score and fit time
    print('Cross validation scores of different parameter combinations:')
    display(results_df[['param_' + i for i in list(search_space.keys())] + ['mean_test_score', 'converted_mean_fit_time']])
    return results_df

# Compute train and test scores
def fun_scores(model, X_train, y_train, cv=5, X_test=None, y_test=None, train_data=None):

    # Get CV train scores of a grid search model
    if (hasattr(model, 'best_score_')):
        MAPE = - np.round(model.best_score_, 6) * 100
        RMSE = np.round(mean_squared_error(y_true=y_train, y_pred=model.predict(X_train), squared=False), 4)
        
        # Print train scores
        print('CV MAPE train data:  {} %'.format(MAPE))
        print('CV RMSE train data: ', RMSE)

        # Show best parameter combination
        print('\nBest model / parameter combination:')
        if (len(model.get_params()) <= 10): display(model.best_estimator_)
        else: display(model.best_params_)

        return {'MAPE': MAPE, 'RMSE': RMSE}

    # Compute CV train scores if model is a usual estimator and measure CV computation time
    else:
        start = time.time()
        cv_scores = cross_validate(estimator=model, X=X_train, y=y_train, cv=cv,
                                   scoring=['neg_mean_absolute_percentage_error', 'neg_root_mean_squared_error'],
                                   n_jobs=-1, verbose=False)
        cv_time = fun_convert_time(start=start, end=time.time())

        MAPE = - np.round(cv_scores['test_neg_mean_absolute_percentage_error'].mean(), 6) * 100
        RMSE = - np.round(cv_scores['test_neg_root_mean_squared_error'].mean(), 4)

        # Print train scores
        print('CV MAPE train data:  {} %'.format(MAPE))
        print('CV RMSE train data: ', RMSE)
        print('CV computation time:', cv_time)
    
    # Compute test scores if test set is given
    if (X_test is not None) & (y_test is not None):

        # Fit model on train data and get predictions for test set
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        # Improve predictions: Sum of predicted Shapley values must be equal to the total costs for all instances
        if (train_data is not None):
            # Assign the predictions to the train_data data frame, which contains all instances and customers with all features and the true Shapley values
            train_data_pred = train_data.assign(Predictions=pd.Series(data=pred, index=X_test.index))

            # Sum up all Shapley values (costs) in the train set (X_train) of each instance to compute the remaining costs, which should be equal to the sum of predictions
            costs_in_X_train = train_data_pred.groupby('Instance ID').apply(lambda X: np.sum(X.loc[[i for i in X.index if i in X_train.index]]['Shapley Value']))
            train_data_pred = pd.merge(left=train_data_pred, right=pd.Series(costs_in_X_train, name='Costs in X_train'), left_on='Instance ID', right_index=True)

            # Compute the remaining costs per instance and the sum of predicted Shapley values
            train_data_pred['Remaining Costs'] = train_data_pred['Total Costs'] - train_data_pred['Costs in X_train']
            train_data_pred['Sum of Predictions'] = train_data_pred.groupby('Instance ID')['Predictions'].transform('sum')

            # Compute new predictions in column 'Improved Predictions' and get all predictions in the order of X_test
            train_data_pred['Improved Predictions'] = train_data_pred['Predictions'] * (train_data_pred['Remaining Costs'] / train_data_pred['Sum of Predictions'])
            #display(train_data_pred[['Instance ID', 'Number Customers', 'Total Costs', 'Costs in X_train', 'Remaining Costs', 'Predictions', 'Sum of Predictions', 'Improved Predictions', 'Shapley Value']])
            pred = pd.Series(data=train_data_pred['Improved Predictions'].dropna(), index=X_test.index)

        # Compute errors
        MAPE_test = np.round(mean_absolute_percentage_error(y_true=y_test, y_pred=pred), 6) * 100
        RMSE_test = np.round(mean_squared_error(y_true=y_test, y_pred=pred, squared=False), 4)

        # Update scores
        MAPE_train, RMSE_train = MAPE, RMSE
        MAPE = {'Train data': MAPE_train, 'Test data': MAPE_test}
        RMSE = {'Train data': RMSE_train, 'Test data': RMSE_test}

        # Print test scores
        print('\nMAPE test data: {} %'.format(MAPE_test))
        print('RMSE test data: {}'.format(RMSE_test))

        # Compute error measures for each instance size group individually
        # Group X by instance size and apply for each group the error measure fct. Use indices of each group to select the regarding true y values and the improved predictions in train_data_pred
        MAPE_cat = X_test.groupby(by='Number Customers').apply(lambda group: mean_absolute_percentage_error(y_true=y_test.loc[group.index], y_pred=train_data_pred.loc[group.index, 'Improved Predictions']))
        RMSE_cat = X_test.groupby(by='Number Customers').apply(lambda group: mean_squared_error(y_true=y_test.loc[group.index], y_pred=train_data_pred.loc[group.index, 'Improved Predictions'], squared=False))

        # Round restults and merge them into a data frame. Show data frame
        MAPE_cat = np.round(MAPE_cat, 6) * 100
        RMSE_cat = np.round(RMSE_cat, 4)
        df = pd.DataFrame(data=[MAPE_cat, RMSE_cat], index=['MAPE', 'RMSE'])
        df['Mean'] = [MAPE_test, RMSE_test]
        print('\nMAPE and RMSE on test data per instance size:'), display(df)

        return {'MAPE': MAPE, 'RMSE': RMSE, 'CV computation time': cv_time, 'Scores per instance size': df}

    else: return {'MAPE': MAPE, 'RMSE': RMSE, 'CV computation time': cv_time}



###############################################################################
# FEATURE FUNCTIONS
###############################################################################
# View feature importance of the tree based models
def plot_feature_weights(model, X_train, n_features):

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