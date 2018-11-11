import collections
import math
def Entropy(y):
    if(len(y) == 0):
        return 0.0
    distribution = collections.Counter()
    for yValue in y:
        distribution[yValue] += 1
    numSamples = sum(distribution.values())
    entropy = 0.0
    for value in distribution:
        p = distribution[value] / numSamples
        entropy += -p * math.log2(p)
    return entropy
def SplitOnFeature(x, y, featureIndex):
    # find the split point to use
    min = None
    max = None
    for i in range(len(x)):
        thisValue = x[i][featureIndex]
        if min == None:
            min = thisValue
            max = thisValue
        if thisValue < min:
            min = thisValue
        if thisValue > max:
            max = thisValue
    threshold = (min + max) / 2.0
    # format is [<below thresh>, <equal or above thresh>]
    # and each of those is [x, y] for the samples that meet the criteria
    sampleSubsets = [[[], []], [[],[]]]
    for i in range(len(x)):
        thisValue = x[i][featureIndex]
        if thisValue < threshold:
            subset = sampleSubsets[0]
        else:
            subset = sampleSubsets[1]

        subset[0].append(x[i])
        subset[1].append(y[i])
    return (threshold, sampleSubsets)
def FindBestSplit(x, y, featureSet = None):
    if featureSet == None: # no restriction on features
        featureSet = set()
        for i in range(len(x[0])):
            featureSet.add(i)
    totalEntropy = Entropy(y)
    samples = len(y)
    bestIndex = 0
    bestGain = 0
    bestSubsets = {}
    bestThreshold = None
    for i in range(len(x[0])):
        if i in featureSet: # if we can consider this feature, then do.
            (threshold, subsets) = SplitOnFeature(x,y,i)
            entropyAfterSplit = 0
            for subset in subsets:
                xVal = subset[0]
                yVal = subset[1]
                entropyAfterSplit += (len(xVal)/samples) * Entropy(yVal)
            gain = totalEntropy - entropyAfterSplit
            if gain > bestGain:
                bestIndex = i
                bestGain = gain
                bestSubsets = subsets
                bestThreshold = threshold
    if bestGain == 0:
        return None
    return (bestIndex, bestThreshold, bestSubsets)
class TreeNode(object):
    def __init__(self):
        self.labelDistribution = collections.Counter()
        self.splitIndex = None
        self.threshold = None
        self.children = []
        self.x = []
        self.y = []
    def AddData(self, x, y):
        self.x += x
        self.y += y
        for newY in y:
            self.labelDistribution[newY] += 1
    def GrowTree(self, minToSplit, featureSet = None):
        if len(self.x) < minToSplit:
            return
        splitAnswer = FindBestSplit(self.x, self.y, featureSet)
        if splitAnswer == None:
            return
        (self.splitIndex, self.threshold, splitSubsets) = splitAnswer
        for subset in splitSubsets:
            childNode = TreeNode()
            childNode.AddData(subset[0], subset[1])
            self.children.append(childNode)
            childNode.GrowTree(minToSplit, featureSet)
    def predict(self,x):
        if self.splitIndex != None:
            if x[self.splitIndex] < self.threshold:
                return self.children[0].predict(x)
            else:
                return self.children[1].predict(x)
        return self.labelDistribution.most_common(1)[0][0]
    def visualize(self, depth=1):
        if self.splitIndex == None:
            print(self.labelDistribution)
        else:
            print("Split on: %d" % (self.splitIndex))
            # less than
            for i in range(depth):
                print(' ', end='', flush=True)
            print("< %f -- " % self.threshold, end='', flush=True)
            self.children[0].visualize(depth+1)
            # greater than or equal
            for i in range(depth):
                print(' ', end='', flush=True)
            print(">= %f -- " % self.threshold, end='', flush=True)
            self.children[1].visualize(depth+1)
class DecisionTree(object):
    """A learns simple decision trees."""
    def __init__(self):
        self.treeNode = TreeNode()
    def fit(self, x, y, featureSet = None, minToSplit = 100, logProgress=False):
        self.treeNode.AddData(x,y)
        self.treeNode.GrowTree(minToSplit, featureSet)
    def predict(self, x):
        y = []

        for example in x:
            y.append(self.treeNode.predict(example))
        return y

    def visualize(self):
        self.treeNode.visualize()