# Brain signals classification for Kaggle challenges

This project explores and evaluates various preprocessing and classification pipelines for 
brain signal analysis (EEG, EMG, MEG) datasets from Kaggle challenges. The goal is to achieve 
competitive performance on these challenges using custom-designed processing and classification pipelines.

**Project Scope**
- Implement a diverse set of preprocessing techniques suitable for brain signal data.
- Develop and explore different classification pipelines utilizing various machine learning algorithms.
- Benchmark these pipelines against top performers in Kaggle challenges involving brain signals.

**Future Development**
- Add intra-subject performance evaluation (with leave-one-out cross-validation).
- Implement hyperparameter tuning possibilities to further optimize the performance of the pipelines.
- Expand the project to include several dozen preprocessing and classification pipeline combinations.
- Design a command-line interface (CLI) for easy testing and parameter tuning of these pipelines.
- Integrate additional datasets for broader experimentation, including popular choices like BCI Competition
IV 2a and 2b datasets frequently referenced in research literature.

## How to Run it?

### Installing & run it
```
git clone https://github.com/nabilalibou/eeg-clf_sandbox.git
pip install -r requirements.txt
cd eeg-clf_sandbox/project
python3 wcci_2020.py
```

## Dataset available:

### Clinical BCI Challenge WCCI-2020: Motor-Imagery Dataset

Motor Imagery dataset from the [Clinical BCI Challenge WCCI-2020](https://sites.google.com/view/bci-comp-wcci/). It consists of EEG brain imaging data for 
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
The Classics: LDA (+shrinkage), SVM, kNN, linear regression (+shrinkage), MDM.  
Artificial Neural Networks: ShallowConvNet, EEGNet and DNN.  
Spatial filtering (CSP, geodesic filtering etc), Dimension reduction (PCA)
and Feature extraction (ex: for covariance-based classification using Riemannian geometry). 

### Results:

Classification results (selected score metrics) for each subject are displayed on the 
console, reported in a JSON file and plotted on a grouped bar plot like this one:

<p align="center">
<img src="docs\readme_img\patient01_eval.png" width="600" height="450">
</p>

The program also produces a json file <em>'final_report.json'</em> containing the scores 
of each method classification averaged for all patients.  

#### Clinical BCI Challenge WCCI-2020 Dataset results

Top 3 Competition results for Within-subject (accuracy %): 78.44, 74.69, 73.75.

*Note: these results were obtained without having access to the truth in the field, unlike me,
who was able to optimize based on this.*

Top 3 model for classification within subject:

|                    | Patient | P01  | P02  | P03  | P04  | P05  | P06  | P07  | P08  | Avg      |
|--------------------|---------|------|------|------|------|------|------|------|------|----------|
| CSP + TS + PCA +LR | Acc     | 85.0 | 87.5 | 87.5 | 75.0 | 75.0 | 60.0 | 70.0 | 90.0 | **78.7** |
|                    | Kappa   | 0.70 | 0.75 | 0.75 | 0.50 | 0.50 | 0.20 | 0.40 | 0.80 | **0.57** |
| CSP + Log-reg      | Acc     | 80.0 | 85.0 | 87.5 | 75.0 | 75.0 | 65.0 | 67.5 | 75.0 | 76.3     |
|                    | Kappa   | 0.60 | 0.70 | 0.75 | 0.50 | 0.50 | 0.30 | 0.35 | 0.50 | 0.53     |
| Cov + FgMDM        | Acc     | 80.0 | 82.5 | 77.5 | 65.0 | 75.0 | 65.0 | 82.5 | 70.0 | 74.7     |
|                    | Kappa   | 0.60 | 0.65 | 0.55 | 0.30 | 0.50 | 0.30 | 0.65 | 0.40 | 0.49     |


=> The classic Riemannian pipeline consisting of : 
- Computing the spatial covariances
- Projection into the tangent space 
- Variable selection procedure (dimension reduction with Principal Component Analysis).
- Linear Discriminant Analysis classification.   

With the standard CSP (Common Spatial Pattern) procedure beforehand seems to give the 
best results.  

=> The popular pipeline CSP followed by a logistic regression comes second.  

=> Geodesic filtering achieved in tangent space with a Linear Discriminant Analysis applied 
on spatial covariances before a classification with Minimum Distance to Mean gave good 
results as well.
