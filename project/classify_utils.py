"""
Contains the functions to perform cross-validation and classification.
"""
import copy
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import (
    make_scorer,
    get_scorer_names,
    _scorer,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.utils import shuffle
from project.constants import *


def cross_val(clf_dict, X, y, score_dict, n_folds=5, return_train_score=False):
    """
    Use cross_validate instead of cross_val_score because we will use multiple score
    metrics.
    Note that this function can optionally return the training score with the test score.
    Use Kfold instead of stratified as the labels are balances (when giving all
    the trials). Shuffled = True so it will shuffle at the beginning and then no more
    (not like ShuffleSplit) so no data overlap.
    :return:
    """
    available_scorers = get_scorer_names()
    for score, score_val in score_dict.items():
        if score in available_scorers:
            score_dict[score] = score
        else:
            # if it is not already a scorer
            if not isinstance(score_val, _scorer._PredictScorer):
                score_dict[score] = make_scorer(score_val)

    cv = KFold(n_folds, shuffle=True)
    # cv = StratifiedKFold(n_folds, shuffle=True)  # if epoch small
    res_dict = {}
    print(f"{'*' * 10} {n_folds}-folds Cross-validation Results {'*' * 10}")
    for clf_name, clf_value in clf_dict.items():
        res = cross_validate(
            clf_value,
            X,
            y,
            cv=cv,
            scoring=score_dict,
            return_train_score=return_train_score,
            return_estimator=True,
        )
        res_dict[clf_name] = res
        print(f"-> {clf_name}:")
        for cv_key, cv_value in res_dict[clf_name].items():
            if "test" in cv_key:
                avg_score = sum(cv_value) / len(cv_value)
                print(f"{cv_key} : {round(avg_score, 3)}")

    return res_dict


def custom_cross_val(clf_dict, X, y, score_dict, n_folds=5):

    # If no Cross-validation iterators defined:
    # X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True,
    # test_size=0.3) #small dataset

    kf = KFold(n_folds, shuffle=True)  # Kfold nice for small dataset
    # kf = StratifiedKFold(n_folds, shuffle=True)
    res_dict = {}
    print(f"{'*'*10} {n_folds}-folds Custom CV Results {'*'*10}")
    for clf_name, clf_value in clf_dict.items():
        print(f"-> {clf_name}:")
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

        for score_name, value in res_dict[clf_name].items():
            avg_score = sum(value) / len(value)
            print(f"{score_name} : {round(avg_score, 3)}")

    return res_dict


def evaluate(clf_dict, X, y, score_dict, X_eval, y_eval, num_runs=5):
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
    for clf_name, clf_value in clf_dict.items():
        y_acc = np.zeros((X_eval.shape[0]), dtype="float64")
        res_dict[clf_name] = {}
        for run in range(0, num_runs):
            X_, y_ = shuffle(X, y)
            clf_value.fit(X_, y_)
            y_pred = clf_value.predict(X_eval)
            y_acc += y_pred
            # cm = confusion_matrix(y_pred, y_eval, labels=clf_value.classes_)
            # disp = ConfusionMatrixDisplay(
            #     confusion_matrix=cm,
            #     display_labels = clf_value.classes_
            # )
            # disp.plot()
            # plt.show()
            if run == num_runs - 1:
                y_acc /= num_runs
                y_avg = (np.rint(np.squeeze(y_acc))).astype(int)
                print(
                    f"{'%' * 8} Evaluate: Prediction vs Truth for the '{clf_name}' "
                    f"pipeline "
                    f"{'%' * 8}"
                )
                print(f"y_pred:\n{y_avg}")
                print(f"y_eval:\n{y_eval}")
                for score_name, scorer in score_dict.items():
                    res_dict[clf_name][f"eval_{score_name}"] = round(scorer(y_eval, y_avg), 3)
                    print(f"{score_name}: {round(scorer(y_eval, y_avg), 3)}")

    return res_dict
