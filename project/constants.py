

from mne.decoding import CSP, Vectorizer

from sklearn.metrics import (accuracy_score, precision_score, roc_auc_score,
                             cohen_kappa_score)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

from pyriemann.estimation import Covariances, ERPCovariances, XdawnCovariances
from pyriemann.tangentspace import TangentSpace, FGDA
from pyriemann.classification import MDM, FgMDM, KNearestNeighbor
from pyriemann.spatialfilters import CSP as covCSP


default_sample_rate = 512
default_channel_names = ['F3', 'FC3', 'C3', 'CP3', 'P3', 'FCz', 'CPz', 'F4', 'FC4', 'C4',
                         'CP4', 'P4']
montage = "standard_1020"
default_cue_time = 3.0
event_id = {'left-hand': 1, 'right-hand': 2}

# bandpass filter parameters
low_freq = 8
high_freq = 24
# time segment (in seconds) used for the classification
default_t_clf = 3

# classification constants
return_train_score = True

### Scores metrics
# To choose : "https://scikit-learn.org/stable/modules/" \
 # "model_evaluation.html#common-cases-predefined-values"

# Score dict
all_score_dict = {
    'accuracy': accuracy_score,
    'precision': precision_score,
    'roc_auc': roc_auc_score,
    'kappa': cohen_kappa_score
}

## Classification pipeline dict containing all the classification methods

# Riemann classification method reference:
# A. Barachant, S. Bonnet, M. Congedo and C. Jutten, “Classification of covariance
# matrices using a Riemannian-based kernel for BCI applications”, in NeuroComputing,
# vol. 112, p. 172-178, 2013

all_clf_dict = {
    "Vect + KNN": make_pipeline(Vectorizer(), KNeighborsClassifier()),
    "Vect + Log-reg": make_pipeline(Vectorizer(), LogisticRegression(max_iter=100)),
    "Vect + Scale + LR": make_pipeline(Vectorizer(), StandardScaler(),
                                       LogisticRegression(max_iter=100)),
    "Vect + LinSVM": make_pipeline(Vectorizer(), LinearSVC(random_state=0)),
    "Vect + kerSVM": make_pipeline(Vectorizer(), SVC()),
    "Vect + LDA": make_pipeline(Vectorizer(), LinearDiscriminantAnalysis()),
    "Vect + RegLDA": make_pipeline(
        Vectorizer(),
        LinearDiscriminantAnalysis(shrinkage='auto', solver='eigen')
    ),
    "CSP + KNN": make_pipeline(CSP(n_components=4, log=True), KNeighborsClassifier()),
    "CSP + Log-reg": make_pipeline(CSP(n_components=4, log=True),
                                   LogisticRegression(max_iter=100)),
    "RegCSP + Log-reg": make_pipeline(CSP(n_components=4, reg='ledoit_wolf', log=True),
                                      LogisticRegression()),
    "CSP + LinSVM": make_pipeline(CSP(n_components=4, log=True),
                                  LinearSVC(random_state=0)),
    "CSP + kerSVM": make_pipeline(CSP(n_components=4, log=True), SVC()),
    "CSP + LDA": make_pipeline(CSP(n_components=4, log=True),
                               LinearDiscriminantAnalysis()),
    "CSP + RegLDA": make_pipeline(
        CSP(n_components=4, log=True),
        LinearDiscriminantAnalysis(shrinkage='auto', solver='eigen')
    ),
    "Cov + MDM": make_pipeline(Covariances(),
                               MDM(metric=dict(mean='riemann', distance='riemann'))),
    "Cov + FgMDM": make_pipeline(Covariances(),
                                 FgMDM(metric=dict(mean='riemann', distance='riemann'))),
    "Cov + TSLR": make_pipeline(Covariances(), TangentSpace(),
                                LogisticRegression()),
    "Cov + TSkerSVM": make_pipeline(Covariances(), TangentSpace(),
                                    SVC()),
    "Cov + TSLDA": make_pipeline(Covariances(), TangentSpace(),
                                 LinearDiscriminantAnalysis()),
    "CSP + TSLR": make_pipeline(Covariances(), covCSP(nfilter=4, log=False),
                                TangentSpace(), LogisticRegression()),
    "CSP + TS + PCA + LR": make_pipeline(Covariances(), covCSP(nfilter=4, log=False),
                                         TangentSpace(), PCA(n_components=2),
                                         LogisticRegression()),
    "ERPCov + TS": make_pipeline(ERPCovariances(estimator='oas'), TangentSpace(),
                                 LogisticRegression()),
    "ERPCov + MDM": make_pipeline(ERPCovariances(estimator='oas'), MDM()),  # default scm, lwf
    "XdawnCov + TS": make_pipeline(XdawnCovariances(estimator='oas'), TangentSpace(),
                                   LogisticRegression()),  # default scm, lwf
    "XdawnCov + MDM": make_pipeline(XdawnCovariances(estimator='oas'), MDM()),
}

# "Cov + TSFGDA": make_pipeline(Covariances(), FGDA(), PCA(n_components=2),
#                              LinearDiscriminantAnalysis()),
# "Cov + TSFGDA + PCA + SVM": make_pipeline(Covariances(), FGDA(), PCA(n_components=2),
#                               SVC()),