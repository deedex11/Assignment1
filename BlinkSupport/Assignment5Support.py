import os
import random
import numpy as np

def LoadRawData(kDataPath, includeLeftEye = True, includeRightEye = True, shuffle=True):
    xRaw = []
    yRaw = []
    
    if includeLeftEye:
        closedEyeDir = os.path.join(kDataPath, "closedLeftEyes")
        for fileName in os.listdir(closedEyeDir):
            if fileName.endswith(".jpg"):
                xRaw.append(os.path.join(closedEyeDir, fileName))
                yRaw.append(1)

        openEyeDir = os.path.join(kDataPath, "openLeftEyes")
        for fileName in os.listdir(openEyeDir):
            if fileName.endswith(".jpg"):
                xRaw.append(os.path.join(openEyeDir, fileName))
                yRaw.append(0)

    if includeRightEye:
        closedEyeDir = os.path.join(kDataPath, "closedRightEyes")
        for fileName in os.listdir(closedEyeDir):
            if fileName.endswith(".jpg"):
                xRaw.append(os.path.join(closedEyeDir, fileName))
                yRaw.append(1)

        openEyeDir = os.path.join(kDataPath, "openRightEyes")
        for fileName in os.listdir(openEyeDir):
            if fileName.endswith(".jpg"):
                xRaw.append(os.path.join(openEyeDir, fileName))
                yRaw.append(0)

    if shuffle:
        random.seed(1000)

        index = [i for i in range(len(xRaw))]
        random.shuffle(index)

        xOrig = xRaw
        xRaw = []

        yOrig = yRaw
        yRaw = []

        for i in index:
            xRaw.append(xOrig[i])
            yRaw.append(yOrig[i])

    return (xRaw, yRaw)


from PIL import Image

def Convolution3x3(image, filter):
    # check that the filter is formated correctly
    if not (len(filter) == 3 and len(filter[0]) == 3 and len(filter[1]) == 3 and len(filter[2]) == 3):
        raise UserWarning("Filter is not formatted correctly, should be [[x,x,x], [x,x,x], [x,x,x]]")

    xSize = image.size[0]
    ySize = image.size[1]
    pixels = image.load()

    answer = []
    for x in range(xSize):
        answer.append([ 0 for y in range(ySize) ])

    # skip the edges
    for x in range(1, xSize - 1):
        for y in range(1, ySize - 1):
            value = 0

            for filterX in range(len(filter)):
                for filterY in range(len(filter)):
                    imageX = x + (filterX - 1)
                    imageY = y + (filterY - 1)

                    value += pixels[imageX, imageY] * filter[filterX][filterY]

            answer[x][y] = value

    return answer

def Featurize(xTrainRaw, xTestRaw, includeGradients=True, includeRawPixels=False, includeIntensities=False):
    # featurize the training data, may want to do multiple passes to count things.
    xTrain = []
    for sample in xTrainRaw:
        features = []

        image = Image.open(sample)

        xSize = image.size[0]
        ySize = image.size[1]
        numPixels = xSize * ySize

        pixels = image.load()

        if includeGradients:
            # average Y gradient strength
            #yEdges = Convolution3x3(image, [[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
            #sumGradient = sum([sum([abs(value) for value in row]) for row in yEdges])
            #count = sum([len(row) for row in yEdges])

            #features.append(sumGradient / count)

            # average Y gradient strength in middle 3rd
            #sumGradient = sum([sum([abs(value) for value in row[8:16]]) for row in yEdges])
            #count = sum([len(row[8:16]) for row in yEdges])

            #features.append(sumGradient / count)

            # y-gradient over 3x3 grid division and xgradient with [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
            yGradients = Convolution3x3(image, [[1, 2, 1],[0, 0, 0],[-1, -2, -1]]) # using Sobel y-gradient filter, [1, 2, 1],[0, 0, 0],[-1, -2, -1]]

            # These are the average and min/max of the split 3x3 division grids
            yGradientGrids = SplitIntoGrids(yGradients)
            for part in yGradientGrids:
                # average y-gradient over 3x3 grid division
                average = sum([sum([value for value in row]) for row in part]) / sum([len(row) for row in part])

                features.append(average)

                # min y-gradient over 3x3 grid division
                minimum = min([min(element) for element in part])
                features.append(minimum)

                # max y-gradient over 3x3 grid division
                maximum = max([max(element) for element in part])
                features.append(maximum)

            # This is the histogram part!
            bins = CalculateHistogramBuckets(yGradients)
            for bin in bins:
                # calculate the percentage of things in bin vs overall gradients
                features.append(len(bin) / sum([len(row) for row in yGradients]))


            # y-gradient over 3x3 grid division and xgradient with [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
            xGradients = Convolution3x3(image, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) # using Sobel y-gradient filter, [1, 2, 1],[0, 0, 0],[-1, -2, -1]]

            # These are the average and min/max of the split 3x3 division grids
            xGradientGrids = SplitIntoGrids(xGradients)
            for part in xGradientGrids:
                # average y-gradient over 3x3 grid division
                average = sum([sum([value for value in row]) for row in part]) / sum([len(row) for row in part])

                features.append(average)

                # min y-gradient over 3x3 grid division
                minimum = min([min(element) for element in part])
                features.append(minimum)

                # max y-gradient over 3x3 grid division
                maximum = max([max(element) for element in part])
                features.append(maximum)

            # This is the histogram part!
            bins = CalculateHistogramBuckets(xGradients)
            for bin in bins:
                # calculate the percentage of things in bin vs overall gradients
                features.append(len(bin) / sum([len(row) for row in xGradients]))


        if includeRawPixels:
            for x in range(xSize):
                for y in range(ySize):
                    features.append(pixels[x,y])


        if includeIntensities:
            for x in range(0, xSize, 2):
                for y in range(0, ySize, 2):
                    features.append(pixels[x,y]/255.0)

        xTrain.append(features)

    # now featurize test using any features discovered on the training set. Don't use the test set to influence which features to use.
    xTest = []
    for sample in xTestRaw:
        features = []
        
        image = Image.open(sample)

        xSize = image.size[0]
        ySize = image.size[1]
        numPixels = xSize * ySize

        pixels = image.load()

        if includeGradients:
            # average Y gradient strength
            #yEdges = Convolution3x3(image, [[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
            #sumGradient = sum([sum([abs(value) for value in row]) for row in yEdges])
            #count = sum([len(row) for row in yEdges])

            #features.append(sumGradient / count)

            # average Y gradient strength in middle 3rd
            #sumGradient = sum([sum([abs(value) for value in row[8:16]]) for row in yEdges])
            #count = sum([len(row[8:16]) for row in yEdges])

            #features.append(sumGradient / count)

             # y-gradient over 3x3 grid division, this is x gradient: [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]
            yGradients = Convolution3x3(image, [[1, 2, 1],[0, 0, 0],[-1, -2, -1]]) # using Sobel y-gradient filter, [1, 2, 1],[0, 0, 0],[-1, -2, -1]]

            # These are the average and min/max of the split 3x3 division grids
            yGradientGrids = SplitIntoGrids(yGradients)
            for part in yGradientGrids:
                # average y-gradient over 3x3 grid division
                average = sum([sum([value for value in row]) for row in part]) / sum([len(row) for row in part])

                features.append(average)

                # min y-gradient over 3x3 grid division
                minimum = min([min(element) for element in part])
                features.append(minimum)

                # max y-gradient over 3x3 grid division
                maximum = max([max(element) for element in part])
                features.append(maximum)

            # This is the histogram part!
            bins = CalculateHistogramBuckets(yGradients)
            for bin in bins:
                # calculate the percentage of things in bin vs overall gradients
                features.append(len(bin) / sum([len(row) for row in yGradients]))


            # y-gradient over 3x3 grid division and xgradient with [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
            xGradients = Convolution3x3(image, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) # using Sobel y-gradient filter, [1, 2, 1],[0, 0, 0],[-1, -2, -1]]

            # These are the average and min/max of the split 3x3 division grids
            xGradientGrids = SplitIntoGrids(xGradients)
            for part in xGradientGrids:
                # average y-gradient over 3x3 grid division
                average = sum([sum([value for value in row]) for row in part]) / sum([len(row) for row in part])

                features.append(average)

                # min y-gradient over 3x3 grid division
                minimum = min([min(element) for element in part])
                features.append(minimum)

                # max y-gradient over 3x3 grid division
                maximum = max([max(element) for element in part])
                features.append(maximum)

            # This is the histogram part!
            bins = CalculateHistogramBuckets(xGradients)
            for bin in bins:
                # calculate the percentage of things in bin vs overall gradients
                features.append(len(bin) / sum([len(row) for row in xGradients]))

        if includeRawPixels:
            for x in range(xSize):
                for y in range(ySize):
                    features.append(pixels[x,y])

        if includeIntensities:
            for x in range(0, xSize, 2):
                for y in range(0, ySize, 2):
                    features.append(pixels[x,y]/255.0)

        xTest.append(features)

    return (xTrain, xTest)


import PIL
from PIL import Image

def VisualizeWeights(weightArray, outputPath):
    size = 12

    # note the extra weight for the bias is where the +1 comes from, just ignore it
    if len(weightArray) != (size*size) + 1:
        raise UserWarning("size of the weight array is %d but it should be %d" % (len(weightArray), (size*size) + 1))

    if not outputPath.endswith(".jpg"):
        raise UserWarning("output path should be a path to a file that ends in .jpg, it is currently: %s" % (outputPath))

    image = Image.new("L", (size,size))

    pixels = image.load()

    for x in range(size):
        for y in range(size):
            pixels[x,y] = int(abs(weightArray[(x*size) + y]) * 255)

    image.save(outputPath)

def SplitIntoGrids(gradients):
    # So gradients is [[24 elements] x 24 ]
    # we want to split it up into nine 8x8 grids
    # we should take the 24/8 and grab each until we rech 8x8 then do the same for the next 
    # take the first 8 rows of the gradient, then divide the elments in each row by 3
    lengthOfGrid = int(len(gradients) / 3) # for us its 24 x 24 -> 24/3 -> 8 -> so each grid has 8 rows and 8 columns
    grids = []
    split = 0
    gridPosition = lengthOfGrid
    position = 0
    while split < 3:
        gridData = gradients[position:gridPosition] # this should give me indexes 0 - 7, 8 - 15, 16 - 23 ex: [[24 pixels] x 8]
        # So next what I need is to divide the single array with 24 by the length of the gradient, and that will give me the amount in each array to add for the grid
        grid1 = []
        grid2 = []
        grid3 = []
        rowCounter = 0
        while rowCounter < len(gridData):
            lastPosition = lengthOfGrid
            grid1.append(gridData[rowCounter][0:lastPosition])
            grid2.append(gridData[rowCounter][lastPosition: (lastPosition * 2)])
            grid3.append(gridData[rowCounter][(lastPosition * 2): (lastPosition * 3)])
            rowCounter+=1
        grids.append(grid1)
        grids.append(grid2)
        grids.append(grid3)

        # update counters and positions
        split = split + 1
        position+=lengthOfGrid
        gridPosition+=lengthOfGrid

    return grids; # which is a [[8 elements] x 8] x 3 nested array. 

def CalculateHistogramBuckets(gradients):
    # take absolute value first
    yGradientsAbsolute = [[abs(value) for value in row] for row in gradients]

    bin1 = [] 
    bin2 = []
    bin3 = []
    bin4 = []
    bin5 = []

    # normalize gradients first
    normalizedGradients = []
    for row in yGradientsAbsolute:
        i = 0
        normalized = []
        while i < len(row):
            if max(row) - min(row) == 0:
                normalized.append(0)
            else:
                normalized.append((row[i] - min(row)) / (max(row) - min(row)))
            i+=1
        normalizedGradients.append(normalized)

    for row in normalizedGradients:
        for value in row:
            if 0.0 <= value < 0.2:
                bin1.append(value)
            if 0.2 <= value < 0.4:
                bin2.append(value)
            if 0.4 <= value < 0.6:
                bin3.append(value)
            if 0.6 <= value < 0.8:
                bin4.append(value)
            if value >= 0.8:
                bin5.append(value)
    bins = []
    bins.append(bin1)
    bins.append(bin2)
    bins.append(bin3)
    bins.append(bin4)
    bins.append(bin5)

    return bins 


