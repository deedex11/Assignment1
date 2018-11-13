from prettytable import PrettyTable
import math
import numpy as np
# Open source library: https://pypi.org/project/PrettyTable/ Make sure to have prettytable.py downloaded in the same directory as this file

def __CheckEvaluationInput(y, yPredicted):
    # Check sizes
    if(len(y) != len(yPredicted)):
        raise UserWarning("Attempting to evaluate between the true labels and predictions.\n   Arrays contained different numbers of samples. Check your work and try again.")

    # Check values
    valueError = False
    for value in y:
        if value not in [0, 1]:
            valueError = True
    for value in yPredicted:
        if value not in [0, 1]:
            valueError = True

    if valueError:
        raise UserWarning("Attempting to evaluate between the true labels and predictions.\n   Arrays contained unexpected value. Must be 0 or 1.")

def Accuracy(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    correct = []
    for i in range(len(y)):
        if(y[i] == yPredicted[i]):
            correct.append(1)
        else:
            correct.append(0)

    return sum(correct)/len(correct)

def upperBound95(accuracy, n):
    return accuracy + 1.96 * math.sqrt((accuracy * (1 - accuracy) / n))

def lowerBound95(accuracy, n):
    return accuracy - (1.96 * math.sqrt((accuracy * (1 - accuracy) / n)))

def Precision(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    truePositives = TotalTruePositive(y, yPredicted)
    falsePositives = TotalFalsePositive(y, yPredicted)

    denominator = (truePositives + falsePositives)

    if denominator == 0:
        return 0
    else:
        return truePositives / denominator

def Recall(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    truePositives = TotalTruePositive(y, yPredicted)
    falseNegatives = TotalFalseNegative(y, yPredicted)

    if ((truePositives + falseNegatives) == 0.0):
        return 0.0

    return truePositives / (truePositives + falseNegatives)

def FalseNegativeRate(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    truePositives = TotalTruePositive(y, yPredicted)
    falseNegatives = TotalFalseNegative(y, yPredicted)

    if ((truePositives + falseNegatives) == 0):
        return 0.0

    return falseNegatives / (truePositives + falseNegatives)

def FalsePositiveRate(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    trueNegatives = TotalTrueNegative(y, yPredicted)
    falsePositives = TotalFalsePositive(y, yPredicted)

    if (falsePositives + trueNegatives == 0):
        return 0.0
    else :
        return falsePositives / (trueNegatives + falsePositives)

def ConfusionMatrix(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    matrix = PrettyTable()
    matrix.title = 'Confusion Matrix'
    matrix.field_names = [" ", "Prediction: 1", "Prediction: 0"]
    matrix.add_row(["Actual: 1", TotalTruePositive(y, yPredicted), TotalFalseNegative(y, yPredicted)])
    matrix.add_row(["Actual: 0", TotalFalsePositive(y, yPredicted), TotalTrueNegative(y, yPredicted)])

    return matrix

def TotalFalsePositive(actual, predicted):
    count = 0

    for i in range(len(actual)):
        if(actual[i] == 0 and predicted[i] == 1):
            count+=1

    return count

def TotalFalseNegative(actual, predicted):
    count = 0

    for i in range(len(actual)):
        if(actual[i] == 1 and predicted[i] == 0):
            count+=1

    return count

def TotalTrueNegative(actual, predicted):
    count = 0

    for i in range(len(actual)):
        if(actual[i] == 0 and predicted[i] == 0):
            count+=1

    return count

def TotalTruePositive(actual, predicted):
    count = 0

    for i in range(len(actual)):
        if(actual[i] == 1 and predicted[i] == 1):
            count+=1

    return count

def ExecuteAll(y, yPredicted):
    accuracy = Accuracy(y, yPredicted)
    print(ConfusionMatrix(y, yPredicted))
    print("Accuracy:", accuracy)
    print("Upper Accuracy Bound", upperBound95(accuracy, len(y)))
    print("Lower Accuracy Bound", lowerBound95(accuracy, len(y)))
    print("Precision:", Precision(y, yPredicted))
    print("Recall:", Recall(y, yPredicted))
    print("FPR:", FalsePositiveRate(y, yPredicted))
    print("FNR:", FalseNegativeRate(y, yPredicted))

#use in conjuction with preditWithProbabilities
def assignment2Comparing(yTestPredictedProb, yTestPredicted, yTest, xTestRaw):
    messageToProbability = []
    index = 0
    for prob in yTestPredictedProb:
        messageToProbability.append((xTestRaw[index], prob))
        index+=1

    # Get top 20 total false positives and the messages
    # So first calculate false positives, if it is one save the probability and messaage in a list
    falsePositives = []
    falseNegatives = []
    for i in range(len(yTestPredicted)):
        if(yTest[i] == 0 and yTestPredicted[i] == 1):
            falsePositives.append(messageToProbability[i])

        if(yTest[i] == 1 and yTestPredicted[i] == 0):
            falseNegatives.append(messageToProbability[i])

    print("The number of false positiveS: " + str(len(falsePositives)))
    print("This number of false negativies: " + str(len(falseNegatives)))

    # Now get 20 worst false positives
    sortedFalsePositives = sorted(falsePositives, key=lambda x: x[1])
    print("The worst 20 false positives: " + str(sortedFalsePositives[:20]))

    # Now get 20 worst false negatives
    sortedFalseNegatives = sorted(falseNegatives, key=lambda x: x[1], reverse=True)
    print("The worst 20 false negatives: " + str(sortedFalseNegatives[:20]))

def ProduceROCPoints(yTest, yProbabilityEstimates):
    metrics = [] #'Threshold, Precision, Recall, FalsePositiveRate, FalseNegativeRate \n'
    thresholdRange = np.arange(0.01, 1.01, 0.01)
    print('Threshold, Precision, Recall, FalsePositiveRate, FalseNegativeRate \n')
    for threshold in thresholdRange:
        yPredictions = []
        for probability in yProbabilityEstimates:

            if probability > threshold:
                yPredictions.append(1)
            else:
                yPredictions.append(0)

        text = str(threshold) + ", " \
               + str(Precision(yTest, yPredictions)) \
               + ", " + str(Recall(yTest, yPredictions)) \
               + ", " + str(FalsePositiveRate(yTest, yPredictions)) \
               + ", " + str(FalseNegativeRate(yTest, yPredictions))
        print(text)
        metrics.append(text)
