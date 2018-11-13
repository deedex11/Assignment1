import collections
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import roc_curve


class KNearestNeighbor(object):

    def __init__(self):
        pass

    def fit(self, xTrain, yTrain, k):
        classifier = KNeighborsClassifier(n_neighbors=k)  
        classifier.fit(xTrain, yTrain)  
        self.classifier = classifier

    def predict(self, xTest):
        yPredicted = self.classifier.predict(xTest)  
        yProbEstimates = self.classifier.predict_proba(xTest)[:,1]

        print("Prbability: " + str(yProbEstimates.tolist()))

        return (yPredicted, yProbEstimates)

    def getROCCurve(self, yTest, yProbEstimates):
        fpr, tpr, thresholds = roc_curve(yTest, yProbEstimates)
        print("%s, %s, %s" % (fpr, tpr, thresholds))
