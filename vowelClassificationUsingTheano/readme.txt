This directory contains the following :
-classifier.py : python code that implements a simple Neural Network using Theano for classification task
-python_speech_features-master : directory that contains python code to extract MFCC features from wav files. Wav recordings of vowel "a" and "e" spoken by 
three people : Tauseef, Bas and Gabi were recorded. mfccCreate.py was used to extract MFCC features into .txt files. These txt files were merged to create
different sets of training and test file such as following :

python_speech_features-master/train/combinedTestTrainExamples.txt :  which contains all MFCC values of all the vowels in test and train directories
python_speech_features-master/test/tauseefAllTestExamples.txt : which contains MFCC values of vowels spoken by Tauseef in test directory

Usage of the classifier script and its output :
[tauseef vowelClassificationUsingTheano]$ python classifier.py python_speech_features-master/train/combinedTestTrainExamples.txt python_speech_features-master/test/tauseefAllTestExamples.txt 
Epoch: 0, Training_cost: 1127.7725315, Training_accuracy: 0.6951040750513046
Epoch: 1, Training_cost: 1158.27610046, Training_accuracy: 0.6547933157431838
Epoch: 2, Training_cost: 1158.2761103, Training_accuracy: 0.6547933157431838
Epoch: 3, Training_cost: 1158.2761103, Training_accuracy: 0.6547933157431838
Epoch: 4, Training_cost: 1158.2761103, Training_accuracy: 0.6547933157431838
Epoch: 5, Training_cost: 1158.2761103, Training_accuracy: 0.6547933157431838
Epoch: 6, Training_cost: 1158.2761103, Training_accuracy: 0.6547933157431838
Epoch: 7, Training_cost: 1158.2761103, Training_accuracy: 0.6547933157431838
Epoch: 8, Training_cost: 1158.2761103, Training_accuracy: 0.6547933157431838
Epoch: 9, Training_cost: 1158.2761103, Training_accuracy: 0.6547933157431838
Test_cost: 384.745387405, Test_accuracy: 0.4949698189134809

