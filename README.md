# Brain signals classification sandbox

Repository used to test several preprocessing/classification pipelines on brain signals 
(EEG/EMG/MEG datasets).

## How to Run it?

### Installing
```
git clone https://github.com/Nabil-AL/eeg-clf_sandbox.git
pip install -r requirements.txt
```

### Dataset available (only one currently):

Motor Imagery dataset from the Clinical BCI Challenge WCCI-2020. It consists of EEG brain imaging data for 
10 hemiparetic stroke patients having hand functional disability.  
The signals were recorded with 12 electrodes, sampled at 512 Hz and initially filtered with 0.1 to 100 Hz 
pass-band filter and a notch filter at 50 Hz. [Dataset Link](https://github.com/5anirban9/Clinical-Brain-Computer-Interfaces-Challenge-WCCI-2020-Glasgow)


### classification:

Choose the classification methods and the metric scores you want to use by filling the 
list ```clf_selection``` and ```score_selection``` with keys coming from the dictionary 
```all_clf_dict``` and ```all_score_dict``` available in <em>constants.py</em>.  

For the moment, only a portion of the score metrics (accuracy, precision, roc auc, kappa) 
and classification methods (LDA, SVM, kNN, linear regression with or without spatial 
filtering + covariance-based classification using Riemannian geometry). 

### Results:

Classification results (selected score metrics) for each subject are displayed on the 
console, reported in a JSON file and plotted on a grouped bar plot like this one:

<p align="center">
<img src="docs\readme_img\patient01_eval.png" width="600" height="450">
</p>

### Future improvements:

<em>

+ Implement Hyperparameter tuning via GridSearch for every classification pipeline.


+ Add others dataset to play with.  


+ Add new classification methods:
Convolutional Neural Network (CNN) especially EEGNet 
("EEGNet: A Compact Convolutional Network for EEG-based Brain-Computer Interfaces" by 
Lawhern et. al.)  
Dense Neural Network (DNN) + Riemannian feature (covariance).   
Use functional connectivity instead of covariance for Riemannian classifiers.  
Add the possibility to do some ensembling with the classification pipelines.  
Add others popular classifiers; XGBoost Classifier, Random forest ... 


+ Add docstrings, comments and optimize code.  


+ Add cross-subject mode (train on 8 subjects and predict the labels of the last two).  


+ Add some possibility to plot the epochs/trials and their characteristics 
(ERPs, topomap, PSD).  


+ Have an automatically filled report with the results and plots and write an analysis 
on the performance of each classifier.





