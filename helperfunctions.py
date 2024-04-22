import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib_inline

import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error


#################### HELPERFUNCTIONS ####################

##### DATASET FUNCTIONS #####
# define function to read in data
def fun_load_file(path, name):

    # select current working directory and subfolder to load the files
    current_directory = os.getcwd()
    subfolder_path = path
    file_path = os.path.join(current_directory, subfolder_path, name)

    # load the file
    return pd.read_excel(io=file_path)

# select columns for training
def fun_preprocessing(data):
    columns = [i for i in data.columns]
    extracted_features = ['Unnamed: 0.1', 'Unnamed: 0', 'Shapley Value Cluster', 'SHAPO', 'Percentage_error']
    train_features = [i for i in columns if i not in extracted_features]
    train_data = data[train_features]
    return train_data

# split dataset into features and target (shapley value)
def fun_split_X_y(data):
    X = data[[i for i in data.columns if not i == 'Shapley Value']]
    y = data['Shapley Value']
    return X, y



##### TIME FUNCTIONS #####
# function to stop time and convert seconds to minutes/hours
def fun_convert_time(start, end):
    seconds = int(end - start)
    if seconds < 60:
        computation_time = f'{seconds} sec'
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        computation_time = f'{minutes} min, {remaining_seconds} sec'
    else:
        hours = seconds // 3600
        remaining_minutes = (seconds % 3600) // 60
        computation_time = f'{hours} h, {remaining_minutes} min'
    return computation_time

# measure fit time of a model
def fun_fit_gridsearch_time(model, X_train, y_train):
    start = time.time()
    model.fit(X_train, y_train)
    fit_time = fun_convert_time(start=start, end=time.time())
    return fit_time



##### SCORING FUNCTIONS #####
# compute train score with cross validation
def fun_train_score(model, X_train, y_train, cv=10, return_results=False):
    start = time.time()
    cv_scores = cross_validate(estimator=model, X=X_train, y=y_train, cv=cv,
                               scoring=['neg_mean_absolute_percentage_error', 'neg_root_mean_squared_error'],
                               n_jobs=-1, verbose=False)
    computation_time = fun_convert_time(start=start, end=time.time())
    
    MAPE = - np.round(cv_scores['test_neg_mean_absolute_percentage_error'].mean(), 6) * 100
    RMSE = - np.round(cv_scores['test_neg_root_mean_squared_error'].mean(), 4)
    
    print('  CV MAPE train data:  {} %'.format(MAPE))
    print('  CV RMSE train data: ', RMSE)
    print('  CV computation time:', computation_time)

    if (return_results == True): return MAPE, RMSE, computation_time

# grid search best model
def fun_best_model(grid_search_model, X_train, y_train, view_results_df=False, return_scores=False):

    MAPE = - np.round(grid_search_model.best_score_, 6) * 100
    RMSE = np.round(mean_squared_error(y_true=y_train, y_pred=grid_search_model.predict(X_train), squared=False), 4)

    print('  CV MAPE train data: {} %'.format(MAPE))
    print('  CV RMSE train data: {}'.format(RMSE))
    
    print('\n  Best model / parameter combination:')
    if (len(grid_search_model.best_params_) < 5): print('  ', grid_search_model.best_params_)
    else: display(grid_search_model.best_estimator_)

    # view cv scores of parameter combinations
    if (view_results_df == True):
        results_df = pd.DataFrame(grid_search_model.cv_results_)
        print('\nCross validation scores of all parameter combinations:')
        display(results_df[['params', 'mean_test_score']])
    
    if (return_scores == True): return MAPE, RMSE

# compute test score
def fun_test_score(model, X_test, y_test, print_params=False):
    prediction = model.predict(X_test)
    MAPE = np.round(mean_absolute_percentage_error(y_true=y_test, y_pred=prediction), 6) * 100
    RMSE = np.round(mean_squared_error(y_true=y_test, y_pred=prediction, squared=False), 4)
    
    if (print_params == True): print('Model parameters:')
    if (print_params == True) and (hasattr(model, 'get_params')): display(model.get_params())
    elif (print_params == True): display(model.best_params())

    print('\nMAPE test data: {} %'.format(MAPE))
    print('RMSE test data: {}'.format(RMSE))

# compute error measures for each instance size group
def fun_category_scores(model, X, y, display_df=True):

    # group X by instance size and apply for each group the error measure fct. Use indices of each group to select the regarding true y values and predict y with group
    MAPE = X.groupby(by='Number Customers').apply(lambda group: mean_absolute_percentage_error(y_true=y.loc[group.index], y_pred=model.predict(group)))
    RMSE = X.groupby(by='Number Customers').apply(lambda group: mean_squared_error(y_true=y.loc[group.index], y_pred=model.predict(group), squared=False))

    # round restults and merge them into a data frame
    MAPE = np.round(MAPE, 6) * 100
    RMSE = np.round(RMSE, 4)
    df = pd.DataFrame(data=[MAPE, RMSE], index=['MAPE', 'RMSE'])
    if display_df == True: 
        print('MAPE and RMSE per instance size:')
        display(df)

    return MAPE, RMSE, df



##### FEATURE FUNCTIONS #####
# view top ten absolute feature weights
def fun_feature_weights(model, X_train):
    feature_weights = pd.Series(data=model.coef_, index=X_train.columns)
    print('\nFeature weights: \n{}'.format(np.abs(feature_weights).sort_values(ascending=False)[:10]))
    print('\nBias:', model.intercept_)

    if any(model.coef_ == 0):
        # number of used features
        print('\nNumber of used features:', np.sum(model.coef_ != 0))
        print('Number of not used features:', np.sum(model.coef_ == 0))

        # features with zero weight
        print('\nNot used features: \n{}'.format(feature_weights.index[feature_weights == 0]))

# view feature importance of the tree based models
def plot_feature_importances(model, X_train, all_features=True):
    
    if (all_features == True):
        plt.figure(figsize=(8, 10))
        indizes = range(n_features)
    else:
        # show only the used features
        plt.figure(figsize=(8, 4))
        indizes = np.where(model.feature_importances_ > 0.001)[0]
        n_features = len(indizes)

    plt.barh(range(n_features), model.feature_importances_[indizes], align='center')
    plt.yticks(np.arange(n_features), X_train.columns[indizes])
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')

# function to plot a heatmap with the grid search cv scores (MAPE) of the parameter combinations
def plot_heatmap(scores_list, param_grid, different_scalers=True):
    if (different_scalers == True): scalers_list = [StandardScaler(), MinMaxScaler(), RobustScaler()]
    parameters = list(param_grid.keys())

    # reshape the scores array for heatmap
    if (len(parameters) == 1):
        n_rows = 1
        n_cols = -1
        y_ticklabels = []
        plotsize = (18, 1)

    elif (len(parameters) == 2):
        n_rows = len(list(param_grid.values())[0])
        n_cols = len(list(param_grid.values())[1])
        y_ticklabels = param_grid[parameters[1]]
        y_label = parameters[1]
        plotsize = (18, 6)
    
    else: print('Too many parameters')

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=plotsize)

    # iterate over all scalers
    for scaler, ax in enumerate(axes):
        scores = scores_list[scaler].reshape(n_rows, n_cols) 

        sns.heatmap(scores, annot=True, fmt=".2f", cmap='viridis', xticklabels=param_grid[parameters[0]], yticklabels=y_ticklabels, ax=ax)
        ax.set_xlabel(parameters[0])
        if (len(parameters) !=1): ax.set_ylabel(y_label)
        ax.set_title(scalers_list[scaler])
            
        # annotating each cell with its corresponding score
        for i in range(len(scores)):
            for j in range(len(scores[i])):
                text = ax.text(j + 0.5, i + 0.5, f"{scores[i, j]:.2f}", ha='center', va='center', color='darkgrey')