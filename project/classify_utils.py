"""
Contains the functions to perform cross-validation and classification.
"""

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import shuffle


def cross_val(clf_dict, X, y, score_selection, n_folds = 5, return_train_score=False):
    """
    Use cross_validate instead of cross_val_score because we will use multiple score
    metrics.
    Note that this function can optionally return the training score with the test score.
    Use Kfold because no need of stratified as the labels are balances (when giving all
    the trials). Shuffled = True so it will shuffle at the beginning and then no more
    (not like ShuffleSplit) so no data overlap.
    :return:
    """
    cv = KFold(n_folds, shuffle=True)
    # cv = StratifiedKFold(n_folds, shuffle=True)  # if epoch small
    res_dict = {}
    for clf_name, clf_value in clf_dict.items():
        res = cross_validate(clf_value, X, y, cv=cv, scoring=score_selection,
                             return_train_score=return_train_score,
                             return_estimator=True)
        res_dict[clf_name] = res

    return res_dict


def manual_cross_val(clf_dict, X, y, score_dict, n_folds=5):

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


def evaluate(clf_dict, X, y, score_dict, X_eval, y_eval):
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
        res_dict[clf_name] = {}
        X, y = shuffle(X, y)
        print(y)
        clf_value.fit(X, y)
        y_pred = clf_value.predict(X_eval)
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

    return res_dict
