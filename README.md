# Predicting Shapley values in routing problems with machine learning
This repository represents the implementation part of my master thesis "Predicting Shapley 
values in routing problems with machine learning".

## Abstract of the master thesis
In routing problems, companies and logistics service providers are not only interested 
in identifying the optimal route with the lowest cost, but also in a fair cost allocation 
among the served customers. A highly regarded method for distributing costs is the 
Shapley value with its unique fairness properties. As the highly intensive computation 
of the exact Shapley values limits the method’s applicability to small instances in routing 
problems, this study presents a novel technique for approximating Shapley values using 
machine learning. The approach is based on the generation of problem-specific features 
to capture the underlying structure of the setting. Therefore, a comprehensive numerical 
study using synthetic instances for the traveling salesman problem and the capacitated 
vehicle routing problem is conducted, including a detailed description of the data 
generation process as well as the implementation of the machine learning algorithms. A 
diverse set of models is considered for the experiments in order to determine the most 
suitable ones for each context. The analyses show excellent results by outperforming 
straightforward proxies and state-of-the-art approximation methods from the literature 
for both problem settings, leading to accurate Shapley value predictions for all customers 
within seconds. Additionally, the paper provides economic findings on the primary cost 
factors based on a feature analysis, as well as a study on the computational efficiency 
of the applied models. Importantly, the generalizability of the methodology to further 
operations research contexts is evaluated by applying the technique to a variant of the 
bin packing problem (BPP). The promising results for the BPP show that the approach 
can be effectively applied to other contexts.
 
**Keywords**: Shapley value · Cost allocation · Traveling salesman problem · Capacitated 
vehicle routing problem · Machine learning

## Instructions on how to run the program
In the `main.py` script, you can define one of the routing problems `"TSP"` or `"CVRP"` 
as the `optimization_problem` variable. After choosing a routing problem, run the script. 
It will execute the following notebooks and scripts in this order:

`b1_feature_selection.ipynb`, `a1_linear.ipynb`, `a2_tree.ipynb` (only for TSP), 
`a3_ensembles.ipynb`, `a4_SVM.ipynb`, `a5_NN.ipynb`, `a6_all_models.ipynb`, 
`03_tuning_results/a_tuning_summary.py`, and `03_tuning_results/a_tuning_time.ipynb` 
(only for TSP).

This procedure will automatically **train**, **tune** and **evaluate** all applied 
machine learning models on the dataset of the defined routing problem. Furthermore, this 
generates all relevant **results** and **output files**. Alternatively, you can run each 
model individually for both routing problems in the respective notebook of the model.

In the notebooks `a1_linear.ipynb`, `a2_tree.ipynb`, `a3_ensembles.ipynb`, `a4_SVM.ipynb`, 
and `a5_NN.ipynb` all applied machine learning models are **implemented** and **tuned**. 
The **best parameter configurations** are stored as a PDF file in the 
`"03_tuning_results"` folder.

The notebook `a6_all_models.ipynb` trains the best configuration of each model on the 
training set and evaluates its performance on the test set. The resulting **test scores** 
are stored as an Excel file in the `"04_test_results"` folder.

All the applied user-defined functions of the notebooks are stored in the 
**ML_functions.py** script.

> ⚠️ Note: Due to storage limitations, the datasets are not included in the repository.<br>
> However, the code to generate them can be found in the `"01_data_generation"` folder.

## Applied machine learning algorithms for the TSP
| Algorithm                                     | Implemented in           |
|-----------------------------------------------|--------------------------|
| K-Nearest Neighbor (KNN)                      | `a1_linear.ipynb`        |
| Linear Regression                             | `a1_linear.ipynb`        |
| Ridge Regression                              | `a1_linear.ipynb`        |
| Polynomial Regression                         | `a1_linear.ipynb`        |
| Decision Tree                                 | `a2_tree.ipynb`          |
| Random Forest                                 | `a3_ensembles.ipynb`     |
| GradientBoostingRegressionTrees (GBRT)        | `a3_ensembles.ipynb`     |
| XGBoost                                       | `a3_ensembles.ipynb`     |
| Linear Support Vector Machine (Linear SVM)    | `a4_SVM.ipynb`           |
| Kernel Machine                                | `a4_SVM.ipynb`           |
| Multilayer Perceptron Neural Network(NN/MLP)  | `a5_NN.ipynb`            |

## Applied machine learning algorithms for the CVRP
| Algorithm                                     | Implemented in           |
|-----------------------------------------------|--------------------------|
| K-Nearest Neighbor (KNN)                      | `a1_linear.ipynb`        |
| Polynomial Regression                         | `a1_linear.ipynb`        |
| Random Forest                                 | `a3_ensembles.ipynb`     |
| XGBoost                                       | `a3_ensembles.ipynb`     |
| Kernel Machine                                | `a4_SVM.ipynb`           |
| Multilayer Perceptron Neural Network(NN/MLP)  | `a5_NN.ipynb`            |

## Applied benchmarks
- SHapley APproximation based on a fixed Order (SHAPO)
- Depot distance

## Author
- [Dennis Götz](https://github.com/dennismgoetz)
