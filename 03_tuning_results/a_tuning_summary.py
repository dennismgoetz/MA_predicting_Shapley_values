import os
import sys
import json
import pickle
import pandas as pd
from fpdf import FPDF

# Set the working directory to the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)

# Add the parent directory to the Python path to load funtions from file ML_funtions
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

# Import helperfunctions
from ML_functions import fun_load_settings, fun_get_features_of_preprocessor

# Define the optimization problem (choose either "TSP" or "CVRP")
default_optimization_problem = "CVRP"

# Call the function to define optimization_problem based on how the notebook is executed
# If the notebook is run by the script "main.ipynb", load optimization_problem from "settings.json". Otherwise use the default optimization problem from above
optimization_problem = fun_load_settings(default_optimization_problem, prints=False)

###############################################################################
# DEFINE MODEL NAMES AND FILE NAMES
###############################################################################

# Create a dictionary with all model names as keys and their corresponding names in a file as values
model_names = {"K-nearest Neighbor (KNN)2": "KNN", 
               "Ridge Regression": "Ridge", 
               "Polynomial Regression": "PR", 
               "Decision Tree": "DT", 
               "Random Forest": "RF", 
               "Gradient Boosting Regression Trees (GBRT)": "GBRT", 
               "Extreme Gradient Boosting (XGBoost)": "XGBoost", 
               "Support Vector Machine (SVM)": "SVM", 
               "Kernel Machine": "KM", 
               "Neural Network (NN)": "NN"}

# Exclude the models that were not tuned for the CVRP problem
if (optimization_problem == "CVRP"):
    for model_name in ["Ridge Regression", "Decision Tree", "Gradient Boosting Regression Trees (GBRT)", "Support Vector Machine (SVM)"]:
        del model_names[model_name]

# Create another dictionary with the three different file categories
file_names = {"Parameter grid/distribution": "param_grid.pkl",
              "Best parameters": "best_params.pkl", 
              "Tuning details": "tuning_details.json"}

# Create a dictionary to store all the tuning details
tuning_details_dict = {}

###############################################################################
# DEFINE FUNCTION to add tuning results to a pdf
###############################################################################
# Function to load all three tuning files for a given model and add the content of each files to the pdf
def fun_pdf_tuning_results(model_key, model_names, file_names, optimization_problem, pdf):
    # Iterate over the three tuning files
    for i, file_key in enumerate(file_names, start=1):

        # Add the dictionary key of the file as subtitle to the pdf
        pdf.set_font("Arial", style="b", size=12)
        pdf.cell(0, 3, "", ln=True) # Add some space on top
        pdf.cell(4, 6, ln=False) # Add some space on the left
        pdf.cell(200, 6, txt=f"  {i}. {file_key}", ln=True, align="left")
        pdf.set_font("Arial", size=12)
        
        # Combine the file name with the subfolder to get the file path
        file_name = str(f"{optimization_problem}_{model_names[model_key]}_{file_names[file_key]}")
        subfolder = "../03_tuning_results/01_files"
        file_path = os.path.join(subfolder, file_name)

        try: 
            # Load the file, depending on whether it is in Pickle or Json format
            if (".pkl" in file_name):
                with open(file_path, "rb") as file:
                    data = pickle.load(file)

                # Get the keys of the parameter grid/distribution dictionary
                if (file_key == "Parameter grid/distribution"): 
                    if (isinstance(data, dict)): keys = data.keys()

                    # Polynomial Regression: grid is a list with two dictionaries (get the keys of the first dictionary)
                    elif (isinstance(data, list)):
                        # Modify the param grid dictionary (create a list with the feature sets and add it to the dictionary instead of "preprocessor")
                        feature_set_list = [fun_get_features_of_preprocessor(data[0]["preprocessor"][0]), fun_get_features_of_preprocessor(data[1]["preprocessor"][0])]
                        data[0].pop("preprocessor")
                        data = {**{"feature_set": feature_set_list}, **data[0]}
                        keys = data.keys()

                # Modify the best parameters dictionary
                if (file_key == "Best parameters") and ("feature_set" in keys):
                    data["feature_set"] = fun_get_features_of_preprocessor(data["preprocessor"]) # Get the used features for polynomial regression
                    data.pop("preprocessor")
                
                # Reorder the dictionary with the best combination
                data = {key: data[key] for key in keys} # Same order as in the parameter grid/distribution dictionary

            else:
                with open(file_path, "r") as file:
                    data = json.load(file)
                    # Add the tuning details to the dictionary
                    tuning_details_dict[model_key] = data
            
            # Add the content of the file to the pdf
            for param in data:
                pdf.cell(12, 6, ln=False) # Add some space on the left
                pdf.cell(200, 6, txt=f"- {param}: {data[param]}", ln=True, align="left")
        
        except FileNotFoundError:
            pdf.cell(12, 6, ln=False) # Add some space on the left
            pdf.cell(200, 6, txt="Error: File not found :(", ln=True)
    
    # Add some space above the next model
    pdf.cell(0, 4, "", ln=True)

###############################################################################
# CREATE THE PDF AND APPLY fun_pdf_tuning_results
###############################################################################
# Create a PDF object, add a page, set the font and create a headline
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", style="b", size=24)
pdf.cell(200, 20, txt=f"Tuning results for the {optimization_problem}", ln=True, align="C")

# Iterate over all models, display the model name as headline and execute the defined function from above for each model
for model_key in model_names:
    pdf.set_font("Arial", style="b", size=18)
    pdf.cell(200, 6, txt=f"{model_key}", ln=True, align="left")
    fun_pdf_tuning_results(model_key, model_names, file_names, optimization_problem, pdf)

###############################################################################
# SAVE THE PDF FILE AND THE TUNING DETAILS AS EXCEL FILE
###############################################################################
# Save the PDF file
pdf_file_name = str(f"b_{optimization_problem}_tuning_results.pdf")
pdf.output(pdf_file_name)

# Get all the values from the tuning details dictionary
search_types = [dictionary["Search type"] for dictionary in tuning_details_dict.values()]
parameter_combinations = [dictionary["Parameter combinations"] for dictionary in tuning_details_dict.values()]
tuning_times = [dictionary["Total tuning time"] for dictionary in tuning_details_dict.values()]
fit_times = [dictionary["Total tuning fit time"] for dictionary in tuning_details_dict.values()]
prediction_times = [dictionary["Total tuning prediction time"] for dictionary in tuning_details_dict.values()]

# Create a Data Frame with all the tuning details and save it as an excel file
df = pd.DataFrame(data=[search_types, parameter_combinations, tuning_times, fit_times, prediction_times], 
                  index=["Search type", "Parameter combinations", "Total tuning time", "Total tuning fit time", "Total tuning prediction time"], 
                  columns=[model_names[name] for name in list(tuning_details_dict.keys())]) # Get the abbreviations of the model names as columns
excel_file_name = str(f"c_{optimization_problem}_tuning_details.xlsx")
df.to_excel(excel_file_name)

print(f"\nScript 'a_tuning_summary.py' completed!")

# Change working directory back to the parent path
os.chdir(parent_directory)