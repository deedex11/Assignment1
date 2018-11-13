import collections

class NeuralNetModel(object):

    def __init__(self):
        self.weights = []
        self.activations = []
        self.error = []
        pass

    def fit(self, xTrain, yTrain, stepSize=0.1, numHiddenlayer=1, numNodePerHiddenLayer=2, numIteration=1, presetWeights=presetWeights):
        self.weights = presetWeights

        #For each iteration:
        #For each training sample:
        #Pass the sample through the network to get activations
        #Propagate error from output layer back through the network
        #Update all the weights


    def predict(self, x):
        return [self.prediction for example in x]

    def loss(self, y, yPredicted):
        squares = []
        for i in range(len(y));
            squares.append(sqrt(yPredicted[i] - y[i]))

        return .5 * sum(squares)