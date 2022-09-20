



## Hyperparameter tuning
# https://www.datasklr.com/select-classification-methods/support-vector-machines


# Parameter tuning with GridSearchCV

#######################
### K-Nearest Neighbors
#######################
# estimator_KNN = KNeighborsClassifier(algorithm='auto')
# parameters_KNN = {
#     'n_neighbors': (1, 10, 1),
#     'leaf_size': (20, 40, 1),
#     'p': (1, 2),
#     'weights': ('uniform', 'distance'),
#     'metric': ('minkowski', 'chebyshev'),
# }
#     # with GridSearch
#     grid_search_KNN = GridSearchCV(
#     estimator=estimator_KNN,
#     param_grid=parameters_KNN,
#     scoring='accuracy',
#     n_jobs=-1,
#     cv=5
# )
# #Parameter setting that gave the best results on the hold out data.
# print(grid_search_KNN.best_params_ )
#
# # Parameter tuning with GridSearchCV
#
# #######################
# ### Support Vector Machines
# #######################
#
# estimator_SVM = SVC(gamma='scale')
# parameters_SVM = {
#     'C': (0.1, 15.0, 0.1),
#     'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
#     'coef0': (0.0, 10.0, 1.0),
#     'shrinking': (True, False),
#
# }
# # with GridSearch
# grid_search_SVM = GridSearchCV(
#     estimator=estimator_SVM,
#     param_grid=parameters_SVM,
#     scoring='accuracy',
#     n_jobs=-1,
#     cv=5
# )
# #Parameter setting that gave the best results on the hold out data.
# print(grid_search_SVM.best_params_)


### PCA
"""
https://github.com/d-bi/software-control/blob/master/playground.ipynb
"""

# Hyperparameter optimization using grid search (Demo from sklearn)
#
# """
# Two generic approaches to parameter search are provided in scikit-learn: for given values,
# GridSearchCV exhaustively considers all parameter combinations, while RandomizedSearchCV
# can sample a given number of candidates from a parameter space with a specified
# distribution. Both these tools have successive halving counterparts HalvingGridSearchCV and
# HalvingRandomSearchCV, which can be much faster at finding a good parameter combination.
# """
# # Exhaustive Grid Search
# param_grid = [
#   {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
#   {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
#  ]
#
# # Randomized Parameter Optimization
# """
# Specifying how parameters should be sampled is done using a dictionary. Additionally, a
# computation budget, being the number of sampled candidates or sampling iterations, is
# specified using the n_iter parameter. For each parameter, either a distribution over
# possible values or a list of discrete choices (which will be sampled uniformly) can be
# specified.
# For continuous parameters, such as C above, it is important to specify a continuous
# distribution to take full advantage of the randomization.
# """
# {'C': scipy.stats.expon(scale=100), 'gamma': scipy.stats.expon(scale=.1),
#   'kernel': ['rbf'], 'class_weight':['balanced', None]}
#
# # Halving and choosing min_resources + number of candidates
# """
# number of resources that is used at each iteration depends on the min_resources
# parameter. If you have a lot of resources available but start with a low number of
# resources, some of them might be wasted (i.e. not used).
# Here, The search process will only use 80 resources at most, while our maximum amount of
# available resources is n_samples=1000. Here, we have min_resources = r_0 = 20.
# For HalvingGridSearchCV, by default, the min_resources parameter is set to ‘exhaust’.
# This means that min_resources is automatically set such that the last iteration can use
# as many resources as possible, within the max_resources limit.
# """
# from sklearn.datasets import make_classification
# from sklearn.svm import SVC
# from sklearn.experimental import enable_halving_search_cv  # noqa
# from sklearn.model_selection import HalvingGridSearchCV
# param_grid= {'kernel': ('linear', 'rbf'),
#              'C': [1, 10, 100]}
# base_estimator = SVC(gamma='scale')
# X, y = make_classification(n_samples=1000)
# sh = HalvingGridSearchCV(base_estimator, param_grid, cv=5,
#                          factor=2, min_resources=20).fit(X, y)
#
# sh.best_estimator_
# sh.n_resources_
# sh = HalvingGridSearchCV(base_estimator, param_grid, cv=5,
#                          factor=2, min_resources='exhaust').fit(X, y)
# sh.n_resources_
#
#
# # for linear svm
# """
# A continuous log-uniform random variable is available through loguniform. This is a
# continuous version of log-spaced parameters. For example to specify C above,
# loguniform(1, 100) can be used instead of [1, 10, 100] or np.logspace(0, 2, num=1000).
# """
# from sklearn.utils.fixes import loguniform
# param_grid_linear_svm = { 'linearsvc__C' : np.logspace(-4, 3, 15)}
# #param_grid_linear_svm = { 'linearsvc__C' : loguniform(1e-4, 1e-3)} #linearsvc need to be list or np array
# # lda, auto shrinkage performs pretty well mostly
# """
# Shrinkage is a form of regularization used to improve the estimation of covariance
# matrices in situations where the number of training samples is small compared to the
# number of features. In this scenario, the empirical sample covariance is a poor
# estimator, and shrinkage helps improving the generalization performance of the classifier.
# Shrinkage LDA can be used by setting the shrinkage parameter of the LinearDiscriminantAnalysis
# class to ‘auto’. This automatically determines the optimal shrinkage parameter in an
#  analytic way following the lemma introduced by Ledoit and Wolf [2]. Note that currently
#  shrinkage only works when setting the solver parameter to ‘lsqr’ or ‘eigen’.
# """
# shrinkage = list(np.arange(0,1.01,0.1))
# shrinkage.append('auto')
# param_grid_lda = {'lineardiscriminantanalysis__shrinkage': shrinkage}
# #n_jobs = None  # for multicore parallel processing, set it to 1 if cause memory issues, for full utilization set to -1
# grids_linear_svm= GridSearchCV(clf["CSP + LinSVM"],
#                             param_grid=param_grid_linear_svm, scoring=scorer)
# grids_linear_svm_auc= GridSearchCV(clf["CSP + LinSVM"],
#                             param_grid=param_grid_linear_svm, scoring='roc_auc')
#
# grids_lda = GridSearchCV(clf["CSP + LDA"],
#                         param_grid=param_grid_lda, scoring=scorer)
#
# grids_linear_svm.fit(x_train, y_train)
# grids_linear_svm_auc.fit(x_train, y_train)
#
# print('LinearSVM: Maximum Cross Validation Score = ',
#       round(grids_linear_svm.best_score_, 3))
# print('LinearSVM: Maximum Cross Validation Score = ',
#       round(grids_linear_svm_auc.best_score_, 3))
#
# grids_lda.fit(x_train, y_train)
# print('LDA: Maximum Cross Validation Score = ', round(grids_lda.best_score_, 3))

