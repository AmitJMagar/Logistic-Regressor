"""
Author: Amit Jagannath Magar
Version:1.0
"""

import csv
import math
from matplotlib import pyplot as plot

def readcsv(filename):
    """
    Function to Read CSV File
    :param filename:
    :return: List of Input attributes X1,X2, and output attribute Y
    """
    x1 = []
    x2 = []
    y=[]
    with open(filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile,delimiter=",")
        num = 0;
        for row in csvReader:
            if (num == 0):
                num = num + 1
                continue
            x1.append(float(row[0]))
            x2.append(float(row[1]))
            y.append(int(row[2]))
    return x1, x2, y

def predict_calc(x1, x2, w0, w1, w2):
    """
    applies sigmoid function on input
    :param x1: Attribute 1
    :param x2: Attribute 2
    :param w0: Weight 1 (Bias)
    :param w1: Weight 2 
    :param w2: Weight 3
    :return: 
    """
    return (1 / (1 + math.exp(-(w0 + x1 * w1 + x2 * w2))))

def plotGraph(sum_of_square_diff, iterations):
    """
    This function plots the graph

    :param sum_of_square_diff: List of sum of squared difference values
    :param iterations: Number of Epochs
    :return:Na
    """
    plot.figure("Some of Square Error")
    plot.plot(iterations, sum_of_square_diff)
    plot.xlabel('Epoch')
    plot.ylabel('Sum of Square Errors (SSE) ')
    plot.show()

def sum_of_difference(predicted,actual):
    """
    This function calculates sum of square difference between actual and predicted values
    for given input
    :param predicted: Predicted Value list
    :param actual: Actual Value list
    :return: List of Sum of Square Difference values
    """
    sum=0
    for i in range(len(predicted)):
        sum+=math.pow((predicted[i]-actual[i]),2)
    return sum;


def csv_write(w0,w1,w2):
    """
    This function computed weights in CSV file
    :param w0: Weight 1 (Bias)
    :param w1: Weight 2
    :param w2: Weight 3
    :return:
    """
    with open('weights.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['w0', 'w1', 'w2'])
        csvwriter.writerow([w0, w1, w2])
    csvfile.close()

def update_weight(x1,x2,y,val,w0,w1,w2):
    """

    :param x1: List of Attribute No 1 values
    :param x2: List of Attribute No 2 values
    :param y: List of Output Attribute values
    :param var: Predicted Value using current weights
    :param w0: Weight 1 (Bias)
    :param w1: Weight 2
    :param w2: Weight 2
    :return:
    """

    alpha=0.02 #Learning Rate

    w0 = w0 + alpha * (y - val) * (val * (1 - val))
    w1 = w1 + alpha * (y - val) * (val * (1 - val)) * x1
    w2 = w2 + alpha * (y - val) * (val * (1 - val)) * x2

    return w0,w1,w2


def plot_xy(x1,x2,y,w0,w1,w2):
    """
    This function plots input points and decision boundry using given input and calculated weights

    :param x1: List of Attribute No 1 values
    :param x2: List of Attribute No 2 values
    :param y: List of Output Attribute values
    :param w0: Weight 1 (Bias)
    :param w1: Weight 2
    :param w2: Weight 2
    :return:
    """
    class_1_x = []
    class_1_y = []
    class_0_x = []
    class_0_y = []

    for i in range(len(x1)):
        if y[i]==0:
            class_0_x.append(x1[i])
            class_0_y.append(x2[i])
        else:
            class_1_x.append(x1[i])
            class_1_y.append(x2[i])
    plot.figure("Decision Boundry")
    plot.scatter(class_0_x, class_0_y, c='b', label='Class 0')
    plot.scatter(class_1_x, class_1_y, c='r', label='Class 1')

    x_1=[]
    min_=int(min(x1))
    max_=int(max(x1))
    for i in range(min_,max_,1):
        x_1.append(i)


    y_2=[]
    for i in range(len(x_1)):
        y_2.append(-(w0 + (w1 * x_1[i])) / w2)
    plot.plot(x_1, y_2, c='k', label='Decision Boundry')
    plot.xlabel('Independant Variable 1')
    plot.ylabel('Independant Variable 2')
    plot.legend()
    plot.show()

def print_table(x1,x2,y,w0,w1,w2):
    """
    This function prints information about the logistic regression model, printing
    classification results using provided input and calculated weights.

    :param x1: List of Attribute No 1 values
    :param x2: List of Attribute No 2 values
    :param y: List of Output Attribute values
    :param w0: Weight 1 (Bias)
    :param w1: Weight 2
    :param w2: Weight 2
    :return:
    """
    class_1 = 0
    class_0 = 0
    miss_class_1 = 0
    miss_class_0 = 0
    predict_val = []
    for i in range(len(x1)):
        predict_val.append((1, 0)[predict_calc(x1[i], x2[i], w0, w1, w2) < 0.5])
        if(y[i]==0 ):
            class_0+=1
            if(predict_val[i]==1):
                miss_class_0+=1
        if (y[i] == 1):
            class_1 += 1
            if (predict_val[i] == 0):
                miss_class_1 += 1
    print("Number inputs in Class 0 : " + str(class_0))
    print("Number of correctly classified Class 0 inputs are : "+str(class_0-miss_class_0))
    print("Number of incorrectly identified Class 0 inputs are : " + str( miss_class_0))
    print("Number inputs in Class 1 : " + str(class_1))
    print("Number of correctly identified class 0 inputs are : " + str(class_1 - miss_class_1))
    print("Number of correctly identified class 0 inputs are " + str(miss_class_1))
    accuracy = ((class_0 - miss_class_0) + (class_1 - miss_class_1)) / len(y)
    print("Accuracy for prediction is : " + str(accuracy * 100))


def logistic_regression():
    """
    This controls over all flow of program calling different functions to perform logistic regression 
    and ploting the decision boundry 
    :return: 
    """
    

    x1,x2,y=readcsv(input("Input the file name"))
    #Intiailizing weight values
    w0=1.0
    w1=0.0
    w2=0.0
    
    iteration=1000 #Epoch values for iteration

    sum_square_diff=[]
    epoch=[]
    
    for eps in range(iteration):
        predict_val=[]
        for iter in range(len(x1)):
            predict_val.append(predict_calc(x1[iter], x2[iter], w0, w1, w2))
            w0, w1, w2 = update_weight(x1[iter], x2[iter], y[iter], predict_val[iter], w0, w1, w2)
        sum_square_diff.append(sum_of_difference(predict_val, y))
        epoch.append(eps)

    #Plot the graph for sum of square differece
    plotGraph(sum_square_diff, epoch)
    csv_write(w0, w1, w2)
    plot_xy(x1,x2,y,w0,w1,w2)
    print_table(x1, x2, y, w0, w1, w2)


if __name__ == '__main__':
    logistic_regression()
