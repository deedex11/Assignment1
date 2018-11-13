from PIL import Image
import Assignment5Support
import numpy as np

image = Image.open('/Users/Mims/Documents/school/assignments/BlinkSupport/dataset_B_Eye_Images/openRightEyes/Oscar_DLeon_0001_R.jpg')

yGradients = Assignment5Support.Convolution3x3(image, [[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
xGradients = Assignment5Support.Convolution3x3(image, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

# y-gradient 9 grids of 8x8 pixels
#yFeatures = Assignment5Support.CalculateGradientFeatures(yGradients)
#print (yFeatures[:5])

# x-gradient 9 grids of 8x8 pixels
#xFeatures = Assignment5Support.CalculateGradientFeatures(xGradients)
#print(xFeatures[:5])

# y-graident 5-bin histogram
yFeatures = Assignment5Support.CalculateHistogramBuckets(yGradients)
yhisto = []
for bin in yFeatures:
    # calculate the percentage of things in bin vs overall gradients
    yhisto.append(len(bin) / sum([len(row) for row in yFeatures]))
print (yhisto[:5])

# x-gradient 5-bin histogram
xFeatures = Assignment5Support.CalculateHistogramBuckets(xGradients)
xhisto = []
for bin in xFeatures:
    # calculate the percentage of things in bin vs overall gradients
    xhisto.append(len(bin) / sum([len(row) for row in xFeatures]))
print (xhisto[:5])
