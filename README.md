# Brain signals classification sandbox

Repository used to test and explore several preprocessing/classification pipelines on 
brain signals (EEG/EMG/MEG datasets). Eventually, several dozen preprocessing pipeline and 
classification pipeline combinations will be available. It will then be possible to test 
them as well as to play with their parameters and hyperparameters through a command line
interface.

## How to Run it?

### Installing & run it
```
git clone https://github.com/Nabil-AL/eeg-clf_sandbox.git
pip install -r requirements.txt
cd eeg-clf_sandbox/project
python3 main.py
```

### Dataset available (only one currently):

Motor Imagery dataset from the Clinical BCI Challenge WCCI-2020. It consists of EEG brain imaging data for 
10 hemiparetic stroke patients having hand functional disability.  
The signals were recorded with 12 electrodes, sampled at 512 Hz and initially filtered with 0.1 to 100 Hz 
pass-band filter and a notch filter at 50 Hz. [Dataset Link](https://github.com/5anirban9/Clinical-Brain-Computer-Interfaces-Challenge-WCCI-2020-Glasgow)  
The training files of each subject contains two variables "rawdata" and "labels":  
The variable "rawdata" is 3-D matrix 80x12x4096 with the format "noOfTrial X noOfChannels X noOfSamples".   
The variable "label" is an 1-D array of dimension 80x1 containing the labels for individual trials in the training data.
Label '1' correspond to the class "right motor attempt" and label 2 is "left motor attempt".

### classification:

Choose the classification methods and the metric scores you want to use by filling the 
list ```clf_selection``` and ```score_selection``` with keys coming from the dictionary 
```all_clf_dict``` and ```all_score_dict``` available in <em>constants.py</em>.  

For the moment, only a portion of the score metrics and classification methods are 
available:  

<ins>Score metrics</ins>: accuracy, precision, roc auc, Cohen’s kappa.  

<ins>Classification methods</ins>:  
Classics ones: LDA (+shrinkage), SVM, kNN, linear regression (+shrinkage), MDM.
Artificial neural networks: ShallowConvNet, EEGNet and DNN.  
Possibilities of spatial filtering (CSP, geodesic filtering etc) or dimension reduction (PCA)
and using others features like covariance (ex: covariance-based classification using Riemannian geometry). 

### Results:

Classification results (selected score metrics) for each subject are displayed on the 
console, reported in a JSON file and plotted on a grouped bar plot like this one:

<p align="center">
<img src="docs\readme_img\patient01_eval.png" width="600" height="450">
</p>

#### Clinical BCI Challenge WCCI-2020 Dataset baseline results

Top 3 model for classification within subject:

|                    | Patient | P01  | P02  | P03  | P04  | P05  | P06  | P07  | P08  | Avg  |
|--------------------|---------|------|------|------|------|------|------|------|------|------|
| CSP + TS + PCA +LR | Acc     | 85.0 | 87.5 | 87.5 | 75.0 | 75.0 | 60.0 | 70.0 | 90.0 | 78.7 |
|                    | Kappa   | 0.70 | 0.75 | 0.75 | 0.50 | 0.50 | 0.20 | 0.40 | 0.80 | 0.57 |

The classic Riemannian pipeline (spatial covariance + projection into the tangent space 
to be classifier) with the standard CSP procedure gives the best results.

### Future improvements:

<em>

+ Implement Hyperparameter tuning possibilities.  


+ Add cross-subject mode (train on 8 subjects and predict the labels of the last two).


+ Add new classification methods: Filter Bank Common Spatial Patterns (FBCSP), some popular
classifiers; XGBoost Classifier, Random forest. Add the possibility to do some ensembling 
with the classification pipelines.


+ add feature like functional connectivity, test feature extraction method like autoencoder ...


+ Add docstrings, comments and optimize code.


+ Add others dataset to play with.  


+ Add some possibility to plot the epochs/trials and their characteristics 
(ERPs, topomap, PSD).  






