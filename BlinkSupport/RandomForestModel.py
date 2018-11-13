import random
import math
import DecisionTreeModel
import collections
import Evaluations
import joblib

class RandomForestModel(object):

    def __init__(self):
        pass

    def fit(self, xTrain, yTrain, xTest, yTest, numTrees, minSplit=2, bagging=True, restrictFeatures=True, featureRestriction=20):
        self.numTrees = numTrees
        self.minSplit = minSplit
        self.bagging = bagging
        self.featureRestriction = featureRestriction

        # Random Forest algorithm psuedocode from slides
        trees = []

        # Not parallel:
        #for i in range(numTrees):
            # this is supposed to bootstrap the sample for eaching training set with bagging (no replacement)
            #if bagging:
                #(xBootstrap, yBootstrap) = self.bootstrap(xTrain, yTrain)
            #else:
                #(xBootstrap, yBootstrap) = xTrain, yTrain
            # this restricts the features the tree is allowed to use when growing
            #featuresToUse = self.randomly_select(featureRestriction, True, len(xTrain[0]))
            # this grows the tree using the set of features
            #trees.append(self.GrowTree(xBootstrap, yBootstrap, featuresToUse, minSplit))

        # Making it paralell!
        self.trees = joblib.Parallel(n_jobs=6)(joblib.delayed(self.individualTree)(xTrain, yTrain, minSplit,
                                                          bagging, restrictFeatures, featureRestriction) for i in range(numTrees))


        #self.trees = trees

    def individualTree(self, xTrain, yTrain, minSplit, bagging, restrictFeatures, featureRestriction):
        # this is supposed to bootstrap the sample for eaching training set with bagging (no replacement)
        if bagging:
            (xBootstrap, yBootstrap) = self.bootstrap(xTrain, yTrain)
        else:
            (xBootstrap, yBootstrap) = xTrain, yTrain
        # this restricts the features the tree is allowed to use when growing
        featuresToUse = self.randomly_select(featureRestriction, restrictFeatures, len(xTrain[0]))
        # this grows the tree using the set of features
        return self.GrowTree(xBootstrap, yBootstrap, featuresToUse, minSplit)

    def predict(self, xTest, threshold=None):
        yPredictions = [self.predict_with_majority(self.trees, xTest[i], threshold) for i in range(len(xTest))]

        # Counts the number of yeses on all the tree the sample got
        yProbabilityEstimates = [self.count_votes(self.trees, xTest[i]) / len(self.trees) for i in range(len(xTest))]
        return (yPredictions, yProbabilityEstimates)

    # this is supposed to bootstrap the sample for eaching training set with bagging (no replacement)
    # returns xbootstrap and yBootstrap
    def bootstrap(self, xTrain, yTrain):
        ySamples = []
        xSamples = []

        while len(xSamples) < len(xTrain):
            index = random.randrange(len(xTrain))
            xSamples.append(xTrain[index])
            ySamples.append(yTrain[index])
        return (xSamples, ySamples)

    # this restricts the features the tree is allowed to use when growing
    # If specified randomly select N features for each tree and restrict the tree to
    # using those (select a different random set for each tree). If set to 0 use all available features.
    # returns features to use
    def randomly_select(self, numToUse, restrictFeatures, featureLength):
        if restrictFeatures:
            randomFeatures = random.sample(range(featureLength), numToUse)
        else:
            randomFeatures = list(range(featureLength))
        return randomFeatures
        # Returns a list of the indexes of the features to use


    # But it also does a mass prediction for each tree on the data
    # this makes an array that contains the predictions for everything with majority on the tree data
    # returns what the majority is for all the predictions
    # your input is [self.predict_with_majority(trees, xTest[i]) for i in len(xTest)]
    def predict_with_majority(self, trees, xRow, threshold=None):

        predictions = []
        for tree in trees:
            prediction = tree.treeNode.predict(xRow, threshold)
            predictions.append(prediction) # TODO: make sure that xRow is ok, its xTest[i]
        # now that we have predictions for every tree, we get the majority and return it as the prediction
        return collections.Counter(predictions).most_common(1)[0][0]

    # Counts the number of yeses on all the tree the sample got
    # returns the number of yes's each tree gave the sample
    # trees, xTest[i]
    def count_votes(self, trees, x):
        yes = 0
        for tree in trees:
            prediction = tree.treeNode.predict(x)
            if (prediction == 1):
                yes+=1
        return yes


    # this grows the tree using the set of features
    # returns a tree
    # xBootstrap, yBootstrap, featuresToUse, minSplit
    def GrowTree(self, xBootstrap, yBootstrap, featuresToUse, minToSplit):
        model = DecisionTreeModel.DecisionTree()
        model.fit(xBootstrap, yBootstrap, featuresToUse, minToSplit, True)
        return model