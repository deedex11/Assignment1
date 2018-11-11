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