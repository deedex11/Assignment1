## Some of this references my answers to previous assignments.
##  Replace my references with references to your answers to those assignments.

## IMPORTANT NOTE !!
## Remember to install the Pillow library (which is required to execute 'import PIL')

import Assignment5Support

## NOTE update this with your equivalent code..
import TrainTestSplit

kDataPath = "/Users/Mims/Documents/school/assignments/BlinkSupport/dataset_B_Eye_Images"

(xRaw, yRaw) = Assignment5Support.LoadRawData(kDataPath, includeLeftEye = True, includeRightEye = True)

(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = TrainTestSplit.TrainTestSplit(xRaw, yRaw, percentTest = .25)

print("Train is %f percent closed." % (sum(yTrainRaw)/len(yTrainRaw)))
print("Test is %f percent closed." % (sum(yTestRaw)/len(yTestRaw)))

print("Calculating features...")
(xTrain, xTest) = Assignment5Support.Featurize(xTrainRaw, xTestRaw, includeGradients=True, includeRawPixels=False, includeIntensities=False)
yTrain = yTrainRaw
yTest = yTestRaw


import Evaluations

######
import MostCommonModel
model = MostCommonModel.MostCommonModel()
model.fit(xTrain, yTrain)
yTestPredicted = model.predict(xTest)
accuracy = Evaluations.Accuracy(yTest, yTestPredicted)
print("Most Common Accuracy: %s with Upper Bound %s and Lower Bound %s" % (accuracy, Evaluations.upperBound95(accuracy, len(yTest)), Evaluations.lowerBound95(accuracy, len(yTest))))

######
import DecisionTreeModel
model = DecisionTreeModel.DecisionTree()
model.fit(xTrain, yTrain, minToSplit=50)
yTestPredicted = model.predict(xTest)
accuracy = Evaluations.Accuracy(yTest, yTestPredicted)
print("Decision Tree Accuracy: %s with Upper Bound %s and Lower Bound %s" % (accuracy, Evaluations.upperBound95(accuracy, len(yTest)), Evaluations.lowerBound95(accuracy, len(yTest))))

######
#import RandomForestModel
#model = RandomForestModel.RandomForestModel()
#model.fit(xTrain, yTrain, xTest, yTest, 10, 50, True, False, 0) # best y gradient
#model.fit(xTrain, yTrain, xTest, yTest, 20, 75, True, False, 0) # best x gradient - numTrees, minSplit=2, bagging=True, restrictFeatures=True, featureRestriction=20
#(yTestPredicted, yProbabilityEstimates) = model.predict(xTest)
#accuracy = Evaluations.Accuracy(yTest, yTestPredicted)
#print("RandomForest Accuracy: %s with Upper Bound %s and Lower Bound %s" % (accuracy, Evaluations.upperBound95(accuracy, len(yTest)), Evaluations.lowerBound95(accuracy, len(yTest))))
#Evaluations.ProduceROCPoints(yTest, yProbabilityEstimates)

##### k means
#import KMeansClustering
#model = KMeansClustering.KMeansClustering(4, 10)
# We want the <x,y> values to be the <firstFeature, secondFeature> so we need to split the x data up into x and y
#xs = [item[0] for item in xTrain]
#ys = [item[1] for item in xTrain]
#model.run(xs, ys, xTrainRaw)

### k nearest neighbor
import KNearestNeighbor
model = KNearestNeighbor.KNearestNeighbor()

for k in [1, 3, 5, 10, 20, 50, 100]:
	print("We are on k = " + str(k))
	model.fit(xTrain, yTrain, k)
	(yTestPredicted, yProbabilityEstimates) = model.predict(xTest)
	accuracy = Evaluations.Accuracy(yTest, yTestPredicted)
	#model.getROCCurve(yTest, yProbabilityEstimates)
	Evaluations.ProduceROCPoints(yTest, yProbabilityEstimates)


##### for visualizing in 2d
#for i in range(500):
#    print("%f, %f, %d" % (xTrain[i][0], xTrain[i][1], yTrain[i]))

##### sample image debugging output

import PIL
from PIL import Image

i = Image.open(xTrainRaw[1])
#i.save("..\\..\\..\\Datasets\\FaceData\\test.jpg")

print(i.format, i.size)

# Sobel operator
xEdges = Assignment5Support.Convolution3x3(i, [[1, 2, 1],[0,0,0],[-1,-2,-1]])
yEdges = Assignment5Support.Convolution3x3(i, [[1, 0, -1],[2,0,-2],[1,0,-1]])

pixels = i.load()

for x in range(i.size[0]):
    for y in range(i.size[1]):
        pixels[x,y] = abs(xEdges[x][y])

#i.save("c:\\Users\\ghult\\Desktop\\testEdgesY.jpg")