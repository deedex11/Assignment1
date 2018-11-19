import Assignment5Support
import numpy as np
import NeuralNetStack
import TrainTestSplit
import Evaluations

kDataPath = "/Users/Mims/Documents/school/assignments/BlinkSupport/dataset_B_Eye_Images"

(xRaw, yRaw) = Assignment5Support.LoadRawData(kDataPath, includeLeftEye = True, includeRightEye = True)

(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = TrainTestSplit.TrainTestSplit(xRaw, yRaw, percentTest = .25)

print("Train is %f percent closed." % (sum(yTrainRaw)/len(yTrainRaw)))
print("Test is %f percent closed." % (sum(yTestRaw)/len(yTestRaw)))

print("Calculating features...")
(xTrain, xTest) = Assignment5Support.Featurize(xTrainRaw, xTestRaw, includeGradients=False, includeRawPixels=False, includeIntensities=True)
yTrain = yTrainRaw
yTest = yTestRaw

#T = [(t0, [1,0,0,0,0]), (t1, [0,1,0,0,0]), (t2, [0,0,1,0,0]), (t3, [0,0,0,1,0]), (t4, [0,0,0,0,1])]

trainingData = []
for i in range(len(xTrain)):
	trainingData.append((xTrain[i], [yTrain[i]]))

#for numHiddenLayer in [1, 2]:
#	for numNodePerHiddenLayer in [2, 5, 10, 15, 20]: 
#		print("Number of hidden layers: " + str(numHiddenLayer))
#		print("Number of nodes per hidden layer: " + str(numNodePerHiddenLayer))
#		if numHiddenLayer == 1:
#			model = NeuralNetStack.NeuralNetStack([len(trainingData[0][0]), numNodePerHiddenLayer, 1], .05, None)
#		else:
#			model = NeuralNetStack.NeuralNetStack([len(trainingData[0][0]), numNodePerHiddenLayer, numNodePerHiddenLayer, 1], .05, None)
#		model.fit(trainingData, 200, xTrain, yTrain, xTest, yTest)
#		(yPrediction, yPredictionProb) = model.predict(xTest)
#		print("This is test set accuracy")
#		Evaluations.ExecuteAll(yTest, yPrediction)
#
#		# Now this is train set accuracy
#		(yPrediction, yPredictionProb) = model.predict(xTrain)
#		print("This is train set accuracy")
#		Evaluations.ExecuteAll(yTrain, yPrediction)

model = NeuralNetStack.NeuralNetStack([len(trainingData[0][0]), 2, 1], .05, None)
model.fit(trainingData, 200, xTrain, yTrain, xTest, yTest)
firstNode = []
secondNode= []
for weights in model.weights[0]:
	firstNode.append(weights[0])
	secondNode.append(weights[1])

Assignment5Support.VisualizeWeights(firstNode, "firstNode.jpg")
Assignment5Support.VisualizeWeights(secondNode, "secondNode.jpg")


