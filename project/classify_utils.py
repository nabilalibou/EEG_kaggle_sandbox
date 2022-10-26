"""
Contains the functions to perform cross-validation and classification.
"""
import copy
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import (make_scorer, get_scorer_names, confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.utils import shuffle
from tensorflow import keras
from project.constants import *


def cross_val(clf_dict, X, y, score_dict, n_folds = 5, return_train_score=False):
    """
    Use cross_validate instead of cross_val_score because we will use multiple score
    metrics.
    Note that this function can optionally return the training score with the test score.
    Use Kfold because no need of stratified as the labels are balances (when giving all
    the trials). Shuffled = True so it will shuffle at the beginning and then no more
    (not like ShuffleSplit) so no data overlap.
    :return:
    """
    # https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values
    available_scorers = get_scorer_names()
    for score, score_val in score_dict.items():
        if score in available_scorers:
            score_dict[score] = score
        else:
            score_dict[score] = make_scorer(score_val)

    cv = KFold(n_folds, shuffle=True)
    # cv = StratifiedKFold(n_folds, shuffle=True)  # if epoch small
    res_dict = {}
    for clf_name, clf_value in clf_dict.items():
        res = cross_validate(clf_value, X, y, cv=cv, scoring=score_dict,
                             return_train_score=return_train_score,
                             return_estimator=True)
        res_dict[clf_name] = res

    return res_dict


def custom_cross_val(clf_dict, X, y, score_dict, n_folds=5):

    # If no Cross-validation iterators defined:
    # X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True,
    # test_size=0.3) #small dataset
    # "https://towardsdatascience.com/" \
    # "train-test-split-and-cross-validation-in-python-80b61beca4b6"
    kf = KFold(n_folds, shuffle=True)  # Kfold nice for small dataset
    # kf = StratifiedKFold(n_folds, shuffle=True)
    res_dict = {}
    for clf_name, clf_value in clf_dict.items():
        res_dict[clf_name] = {}
        cnt_loop = 0
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf_value.fit(X_train, y_train)
            y_pred = clf_value.predict(X_test)
            for score_name, scorer in score_dict.items():
                if cnt_loop == 0:
                    res_dict[clf_name][f"test_{score_name}"] = []
                res_dict[clf_name][f"test_{score_name}"].append(scorer(y_test, y_pred))

            cnt_loop += 1

    return res_dict


def evaluate(clf_dict, X, y, score_dict, X_eval, y_eval, num_epochs=300):
    """
    Train on whole dataset
    (
    "https://datascience.stackexchange.com/questions/33008/"
    "is-it-always-better-to-use-the-whole-dataset-to-train-the-final-model"
    )
    Shuffle train data:
    "https://datascience.stackexchange.com/questions/24511/"
    "why-should-the-data-be-shuffled-for-machine-learning-tasks/24524#24524"
    :param clf_dict:
    :param X:
    :param y:
    :param X_eval:
    :param y_eval:
    :return: res_dict:
    """
    res_dict = {}
    score_list = []
    #for i in range(0, 5):
    for clf_name, clf_value in clf_dict.items():
        print("goo")
        res_dict[clf_name] = {}
        X, y = shuffle(X, y)
        # If the classifier is a Neural Network
        # if (char in clf_name for char in NN_clfs):
        #     print("one time")
        #     #X_reshaped = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        #     X_ev_reshaped = X_eval.reshape(X_eval.shape[0], X_eval.shape[1],
        #                                 X_eval.shape[2], 1)
        #     y_reshaped = y.reshape(-1, 1)
        #     print("fit")
        #     clf_value.fit(X, y)
        #     # clf_value.fit(X, y,  batch_size, num_epochs,
        #     #               validation_data=(X_eval, y_eval), verbose=False)
        #     y_pred = clf_value.predict(X_ev_reshaped)
        #     y_pred = (np.rint(np.squeeze(y_pred))).astype(int)
        # else:
        clf_value.fit(X, y)
        y_pred = clf_value.predict(X_eval)
        print("print")
        print(f"{'%'*8} Evaluate: Prediction vs Truth for the '{clf_name}' pipeline "
              f"{'%'*8}")
        print(f"y_pred:\n{y_pred}")
        print(f"y_eval:\n{y_eval}")

        # cm = confusion_matrix(y_pred, y_eval, labels=clf_value.classes_)
        # disp = ConfusionMatrixDisplay(
        #     confusion_matrix=cm,
        #     display_labels = clf_value.classes_
        # )
        # disp.plot()
        # plt.show()
        for score_name, scorer in score_dict.items():
            res_dict[clf_name][f"eval_{score_name}"] = scorer(y_eval, y_pred)
        #score_list.append(res_dict)

    return res_dict, score_list
