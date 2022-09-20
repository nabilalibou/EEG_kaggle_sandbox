# Clinical Brain Computer Interfaces Challenge WCCI 2020 Glasgow

This is the dataset for the competition "Clinical Brain Computer Interfaces Challenge" to be held at WCCI 2020 at Glasgow. There are the EEG data of 10 hemiparetic stroke patients who are impaired either by left or right hand finger mobility. There are two files for each participant. The file name ending with "T" designates the training file, and the file name ending with "E" designates the evaluation/testing file. For example, the filename "Parsed_P05T" suggests the training file for participant P05, while "Parsed_P05E" suggest the evaluation/testing file for the same participant. The training files contains the labels correspoding to each trial, while the labels for trials of evaluation/testing files are not provided. The objective of the competition is to find the labels corresponing to the trials of the evaluation/testing files. 

## Description of the dataset
Here we are describing the contents of each file in the dataset. All the files are in .mat (MATLAB) format, so it can easily be opened using a MATLAB software. As you open any training file for any participant (for example, file "Parsed_P05T", which is the training file for participant "P05") you will find two variables "rawdata" and "labels". The variable "rawdata" is a 3-D matrix the dimensions of which are in the format "noOfTrial X noOfChannels X noOfSamples". The "noOfTrials" designates how many trials are there in the training file. Here, in all the files noOfTrials=80, which means there are a total of 80 trials in the training file. Next, "noOfChannels" designates how many channels are used during the recording. Here, in all the files noOfChannels=12, which means there were 12 EEG channels during the recording. These channels are according to the 10-20 international system as follows in a serial manner: index1= F3,  index2= FC3,  index3= C3,  index4= CP3,  index5= P3,  index6= FCz,  index7= CPz,  index8= F4,  index9= FC4,  index10= C4, index11= CP4, index12= P4. Next is the noOfSamples which denotes the number of samples in each trial, which in this case is 4096. The explanation for this is that each trial is of 8 s long and the data was recorded with a sampling rate of 512 Hz. So, 8x512 = 4096. Thus rawdata(5, 10, :) contains the information about the activity of EEG channel C4 at 5th trial, rawdata(10, 5, :)  contains the information about the activity of EEG channel P3 at 10th trial and so on. Now, the variable "labels" is an 1-D array of dimension 80x1 containing the labels for individual trials in the training data. For example, label(1) contains the label for trial index 1 and label(80) contains the label for trial index 80. As there are two classes namely "left motor attempt" and "right motor attempt" labels are either '1' or '2', where '1' corresponds to the "right motor attempt" and 2 corresponds to the "left motor attempt". The same explanation goes for the evaluation or testing files with exceptions that the noOfTrials=40, which means the dimension of rawdata will be 40x12x4096 and there would no "labels" as it is to be predicted by the classifier trained on the training data.

| Training Files | Evaluation/Testing Files|
| ------ | ------ |
| Parsed_P01T | Parsed_P01E |
| Parsed_P02T | Parsed_P01E |
| * | * |
| * | * |
| Parsed_P08T | Parsed_P08E |
| Not Given | Parsed_P09E |
| Not Given | Parsed_P10E |

## Submission Guidelines
'Submission' means the material submitted by you in the manner and the format specified on the Website via the Submission form on the Competition Website. The submission will include a 4-page IEEE conferences format paper describing the methods. A link to the open-access software repository were the original code has been uploaded (GitHub, Bitbucket or similar). And a Microsoft Excel file with the predictions for the test data. This file must include the following columns: 1) subject ID, 2) trial index, 3) prediction. The winners of the competition will be based on the kappa value of the predictions with respect to the original labels. An example of submitting the predictions on test data is given as follows:


| Subject Name | Trial index | Prediction (class1=1, class2=2) |
| ------ | ------ | ------ | 
| P01 | 1 | 1 |
| P01 | 2 | 1 |
| * | * | * |
| * | * | * |



Submissions must be received prior to the Competition deadline and adhere to the guidelines for Submissions specified on the Website. The sharing of codes and data privately outside the team is prohibited. The competition host has the right to publicly disseminate any entries or models. The solutions need to be made available under a popular OSI-approved license in order to be eligible for the award/prize. There is no maximum team size. In case of multiple submissions from the same team, the latest submission will be considered for evaluation. The winners of 1) within subject and 2) cross subject challenge will be announced separately. Results from the competition will be announced during the conference and it will be publicized via IEEE channels. For "within subject" prediction use the training file for a particular subject and predict the labels of all the 40 trials given in the "evaluation/testing" file of the same subject. Make table as mentioned above for all the predictions with 3 columns as subject name, trial index, and prediction for all the subjects P01 to P8. Remember that the predictions should be given for all the participants and all the trials. Any partially incomplete submission will be disqualified. Tables must given in "Microsoft Excel file" and separate sheets within the "Microsoft Excel file" should be there for separate subjects predictions. For cross-subject prediction the training data of participant P01 to P08 will have to be used to make a generalised classifier which will be used predict the classess of participant P09 and P10. It is to be noted that the training files for P09 and P10 would not be there. The results of Cross-subject prediction needs to be uploaded in a separate "Microsoft Excel file" file containing separate sheets for subject P09 and subject P10. There will be a submission link in the website, which will redirect to a google form asking for "TeamName", "Name of the team-lead", "Affiliation", "Contact details" and "file upload options" for "within subject" and "cross-subject" results. There would also be a text box to share the link of the github repository where the code must be shared and readme file (*.md) should be there describing the algorithm "how to run" instruction for the code. A PDF file upload option would also be there to upload 4-page IEEE conferences format paper describing the methods. Please note that individuals may only belong to a single team.

## Description of the experiment related to the dataset

The experimental protocol follows a conventional SMR BCI architecture consisting of two stages. The first stage is the data acquisition without giving any feedback, from which an initial classifier is trained using the extracted features. This is followed by the online BCI stage that issues neurofeedback on the basis of the classifier outputs. Data acquisition during first stage consists of two runs of 40 trials and each run takes about 7 min and 30 s. Next, the classifier was trained in the offline mode and it takes nearly 30 s, which is followed by one feedback run of 40 trials in online BCI mode. In each run 20 trials are left and remaining 20 trials are right hand MI. Each trial period in the calibration phase was of 8 s (Shown in the timing diagram Fig. 1), within which first three seconds was the preparatory phase where a message stating "get ready" appeared in the middle of a computer screen. After a 2 s period, a beep sound occurred and then at the end of the 3rd second, a cue in the form of a hand-image appeared on the screen, either on the left or in the right side. According to the appearance of the cue, the participants were instructed to perform an motor-attempt task of a left or right-hand grasp. In the calibration stage, the cue lasted up to the end of 8th second, after which the screen goes blank for a random period of 2 to 3 s before the start of another trial. the scalp EEG was recorded with 12 electrodes covering these areas at F3, FC3, C3, CP3, P3, FCz, CPz, F4, FC4, C4, CP4, and P4 locations according to 10–20 international system. The signals were sampled at 512 Hz and initially filtered with 0.1 to 100 Hz pass-band filter and a notch filter at 50 Hz during data acquisition.



