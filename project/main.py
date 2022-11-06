"""
Script used to test several preprocessing/classification pipelines on brain signals
(EEG/EMG/MEG datasets).
This script is the main.py to run after having select the dataset as well as the
classification pipelines and score metrics from constants.py.
Data are first loaded before going through preprocessing. It is then possible to perform
cross-validation with the different classifiers before evaluating them, plot them and
save their performance in a json file.
"""

from project.constants import *
from project.preprocessing_utils import (load_data_from_mat, butter_bandpass_filter,
                                         prepare_data)
from project.classify_utils import cross_val, custom_cross_val, evaluate
from project.model_utils import *
from project.display_result_utils import (print_results, save_barplot, write_subj_report,
                                          write_final_report)

import os
import mne
from scipy.io import loadmat
import glob

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}, to filter out tf logs

data_path = "./data/"
fig_dir = "./results/"
number_of_subject = 2  # 8
mode = "eval"   # test or eval
doCustomCV = True  # use cross_val() or custom_cross_val()
plot_erp = False
# See constants.py
# clf_selection = ["CSP + Log-reg", "CSP + LDA", "Cov + TSLR", "Cov + TSLDA",
#                  "Cov + FgMDM", "CSP + TS + PCA + LR"]
#clf_selection = ["ERPCov + FgMDM","ERPCov + CSP + TS + PCA + LR", "XdawnCov + FgMDM", "XdawnCov + CSP + TS + PCA + LR"]
clf_selection = ["DNN", "Cov + DNN"]
score_selection = ['accuracy', 'kappa']

clf_dict = {}
for clf in clf_selection:
    for pipeline_name, pipeline_value in all_clf_dict.items():
        if clf == pipeline_name:
            clf_dict[pipeline_name] = pipeline_value

score_dict = {}
for score in score_selection:
    for score_name, scorer in all_score_dict.items():
        if score == score_name:
            score_dict[score_name] = scorer

# train_files = glob.glob(data_path + '/*[1-8]T.mat')
# eval_files = glob.glob(data_path + '/*[1-8]E.mat')
# if not len(train_files) or (len(train_files) != len(eval_files)):
# print(f"Error when fetched the training and evaluation .mat files from {data_path}")

# Get the channel location file if it exists
channel_file = glob.glob(data_path + '/*.locs')
if channel_file:
    if len(channel_file) > 1:
        print("error several .locs files in the folder")
    else:
        print("channel file found")
        montage = mne.channels.read_custom_montage(channel_file[0])
else:
    print("no channel file found")
    montage = default_channel_names

if not os.path.exists(fig_dir):
    try:
        os.mkdir(fig_dir)
        print(f"folder '{fig_dir}' created")
    except OSError:
        print(f"creation of the directory {fig_dir} failed")

results_list = []
for i in range(1, number_of_subject+1):
    if i < 10:
        subj_nbr = f"0{i}"
    else:
        subj_nbr = i
    train_file = str(f"{data_path}parsed_P{subj_nbr}T.mat")
    eval_file = str(f"{data_path}parsed_P{subj_nbr}E.mat")
    print(f"treating subject {i}")

    # get the eeg info if they exist
    if i == 1:
        try:
            # ndarray (1,1)
            cue_time = loadmat(train_file)["cueAt"][0][0]
        except KeyError as e:
            print("No cue time found") #verbose
            cue_time = default_cue_time
        try:
            # ndarray (1,1)
            sample_rate = loadmat(train_file)["sampRate"][0][0]
        except KeyError as e:
            print("No sample rate found")  # verbose
            sample_rate = default_sample_rate
    try:
        raw_train_data, labels_train = load_data_from_mat(train_file)
        raw_eval_data, labels_eval = load_data_from_mat(eval_file)
    except KeyError:
        print("error")
        exit(1)

    raw_train_filtered = butter_bandpass_filter(raw_train_data, low_freq, high_freq,
                                                sample_rate, order=5)
    raw_eval_filtered = butter_bandpass_filter(raw_eval_data, low_freq, high_freq,
                                               sample_rate, order=5)
    #if plot_erp:
        # Transpose EEG data and convert from uV to Volts ?
        # raw_training[:-1] *= 1e-6
        # function plot_erp
    X, y = prepare_data(raw_train_filtered,
                        labels_train,
                        sample_rate,
                        t_low=-default_t_clf)
    X_eval, y_eval = prepare_data(raw_eval_filtered,
                                  labels_eval,
                                  sample_rate,
                                  t_low=-default_t_clf)

    if mode == "test":
        if doCustomCV:
            results_dict_customcv = custom_cross_val(clf_dict, X, y, score_dict, 5)
            results_dict = results_dict_customcv
        else:
            results_dict_cv = cross_val(clf_dict, X, y, score_dict, 5,
                                        return_train_score)
            results_dict = results_dict_cv
    else:
        results_dict_eval = evaluate(clf_dict, X, y, score_dict, X_eval, y_eval)
        results_dict = results_dict_eval

    results_list.append(results_dict)

    ## Results
    write_subj_report(results_dict, f"{fig_dir}patient{subj_nbr}.json")

    if i == number_of_subject:
        write_final_report(results_list, f"{fig_dir}final_report.json")

    scores_results_dict, methods = print_results(subj_nbr,
                                                 results_dict,
                                                 mode,
                                                 return_train_score=False)

    save_barplot(scores_results_dict, methods, subj_nbr, fig_dir, mode)
