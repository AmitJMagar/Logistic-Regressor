# Logistic Regressor
Write a python program logreg.py that will do the following.

(a) Given a .csv file containing a list of samples in a 2D space (one per line; variable values before the target class             value of 0 or 1 in the final column), the program will fit the weight vector w for the linear model using batch training (i.e. iterating over the samples, and then updating weights after classifying each sample), and then save the weight values to a file weights.csv. The program should then use matplotlib to produce a line graph showing epoch (pass over the entire training set) vs. the sum of squared error for all samples in each epoch.

(b) Given a .csv file of samples and a weight vector file, the program produces a plot (using matplotlib) showing the samples with their true class from the file, and plots the decision boundary of the classifier as a line. Use color and/or shape to make clear which samples are from Class 1, and which are from Class 0. The program should then report the number of correctly and incorrectly classified samples for each class.
