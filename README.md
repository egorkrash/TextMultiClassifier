# RNN model which predicts speaker on USA debates by given text

## Structure

1. Model
 * ClassifierProject.ipynb - the main notebook where you can find the solution step by step.
 * model.h5 - saved architecture of my neural network.
 * goodmodel1_weights.h5 - just weights with which I achieved a pretty good accuracy (~0.73).
 
2. App
 * make_prediction.py - script which takes text from **input.txt** and uses it to predict speaker. It also can take text as system argument so you can play with it two ways.
 * predict_for_csv.py - makes predictions for the whole dataset by given path of csv file and the name of column containing texts.
 
