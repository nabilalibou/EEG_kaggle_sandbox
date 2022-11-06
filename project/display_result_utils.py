"""
Contains the functions dedicated to print and plot the results as well as store them
in a .json file
"""

import json
import numpy as np
from matplotlib import pyplot as plt


def print_results(subj_nbr, results_dict, mode="test", return_train_score=False):
    """
    :param subj_nbr:
    :param results_dict:
    :param mode:
    :param return_train_score:
    :return: scores_results_dict, methods:
    """
    print(f"{'='*10} Classification Scores Comparison for Patient {subj_nbr} {mode} "
          f"mode) {'='*10}")
    scores_results_dict = {}
    methods = []
    cnt_loop1 = 0
    for pipeline, scores_value in results_dict.items():
        cnt_loop2 = 0
        methods.append(pipeline)  # methods
        print(f"\n{'*'*8} {pipeline} {'*'*8}")
        for score_name, value in scores_value.items():
            # instantiations
            if cnt_loop1 == 0:
                scores_results_dict[score_name] = {}
            if cnt_loop2 == 0:
                scores_results_dict[score_name][pipeline] = []
            if mode in score_name:
                mean_score = round(np.mean(value), 3)
                print(f"{mode.capitalize()}: {score_name} = {mean_score}")
            if return_train_score and mode in score_name:
                train_mean_score = round(np.mean(value), 3)
                print(f"Train: {score_name} = {train_mean_score}")

            scores_results_dict[score_name][pipeline] = [
                mean_score,
                np.std(value)
            ]
            cnt_loop2 += 1
        cnt_loop1 += 1

    return scores_results_dict, methods


def write_subj_report(results_dict, report_path):

    try:
        report_file = open(report_path, "r")
        try:
            data = json.loads(report_file.read())
        except json.decoder.JSONDecodeError as e:
            data = {}
    except FileNotFoundError:
        data = {}
        pass
    try:
        with open(report_path, 'w') as json_file:
            for pipeline, scores_value in results_dict.items():
                if pipeline not in data:
                    data[pipeline] = {}
                for score_name, value in scores_value.items():
                    data[pipeline][score_name] = round(np.mean(value), 3)
            json.dump(data, json_file, indent=2)
    except TypeError:
        raise TypeError("⚠️  Unable to serialize the object")


def write_final_report(results_list, report_path):

    try:
        with open(report_path, 'w') as json_file:
            subj = 0
            dict_all = {"average": {}}
            for result_dict_subj in results_list:
                subj += 1
                dict_all[f"subject_{subj}"] = {}
                for pipeline, scores_value in result_dict_subj.items():
                    dict_all[f"subject_{subj}"][pipeline] = {}
                    if pipeline not in dict_all["average"]:
                        dict_all["average"][pipeline] = {}
                    for score_name, value in scores_value.items():
                        dict_all[f"subject_{subj}"][pipeline][score_name] = \
                            round(np.mean(value), 3)
                        if score_name not in dict_all["average"][pipeline].keys():
                            dict_all["average"][pipeline][score_name] = 0
                        dict_all["average"][pipeline][score_name] += np.mean(value)
                        if subj == len(results_list):
                            dict_all["average"][pipeline][score_name] = \
                                round(dict_all["average"][pipeline][score_name]/subj, 3)
            json.dump(dict_all, json_file, indent=2)
    except TypeError:
        raise TypeError("⚠️  Unable to serialize the object")


def save_barplot(scores_results_dict, methods, subj_nbr, fig_dir, mode):

    # Group barchart
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html

    x = np.arange(len(methods))
    bar_width = 1.2
    fig, ax = plt.subplots()

    rects_list = []
    counter = 0
    for score_name, pipelines in scores_results_dict.items():
        ax_y = []
        deviation = []
        for pipeline, results in pipelines.items():
            ax_y.append(results[0])
            deviation.append(results[1])
        score_name = score_name.replace(f'{mode}_', '')
        # To space out each x points
        #ax_xx = [width*(len(scores_results_dict)*i + counter) for i in x]
        ax_x = x*(len(scores_results_dict)*(1+bar_width/2))*bar_width+bar_width*counter
        rects_list.append(
            ax.bar(ax_x, ax_y, bar_width, yerr=deviation, label=f"{score_name}",
                   edgecolor='white',capsize=8*bar_width/len(scores_results_dict),
                   alpha=0.5)
        )
        counter += 1

    ax.set_ylabel("Scores")
    ax.set_xticks(ax_x-bar_width*(counter-1)/2, methods)
    ax.legend()
    ax.set_title(f"Scores by classification methods for Patient {subj_nbr}")
    ax.yaxis.grid(True)

    for bar in rects_list:
        ax.bar_label(bar)

    figure_size = plt.gcf().get_size_inches()
    factor = len(methods)/5 + bar_width*len(scores_results_dict)/4
    plt.gcf().set_size_inches(factor * figure_size)
    file_name = f"{fig_dir}patient{subj_nbr}_{mode}.png"
    plt.savefig(file_name)
    print(f"barplot saved at {file_name}")
    # plt.show()