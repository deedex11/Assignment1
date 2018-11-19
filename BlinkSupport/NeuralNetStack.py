from itertools import product
import numpy
import math

def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

def d_sigmoid(y):
    return y * (1 - y)

class NeuralNetStack(object):
    def __init__(self, layer_sizes, alpha=0.7, seed=None): # layer sizes is like [2, 10, 20, 15, 2] # 2 input, 2 output, 10 in hidden 1, 20 in hidden 2...

        self.alpha = alpha
        state = numpy.random.RandomState(seed)
        self.weights = [state.uniform(-0.05, 0.05, size)
                        for size in zip(layer_sizes[:-1], layer_sizes[1:])]

    def _feed_forward(self, x):
        yield x
        for w in self.weights:
            x = sigmoid(numpy.dot(x, w))
            yield x

    def _deltas(self, layers, output):
        delta = d_sigmoid(layers[-1]) * (output - layers[-1])
        for layer, w in zip(layers[-2::-1], self.weights[::-1]):
            yield delta
            delta = d_sigmoid(layer) * numpy.dot(delta, w.T)

    def _learn(self, layers, output):
        deltas = reversed(list(self._deltas(layers, output)))
        return [w + self.alpha * numpy.outer(layer, delta)
                for w, layer, delta in zip(self.weights, layers, deltas)]

    def fit(self, training_data, rounds, xTrain, yTrain, xTest, yTest):
        #for _, (input, output) in product(range(rounds), training_data):
            #layers = self._feed_forward(numpy.array(input))
            #self.weights = self._learn(list(layers), numpy.array(output))
        #print("Iteration, Train set loss, Test set loss")
        for i in range(rounds):
            for index in range(len(training_data)):
                (input, output) = training_data[index]
                layers = self._feed_forward(numpy.array(input))
                self.weights = self._learn(list(layers), numpy.array(output))
            #if i % 5 ==0:
                #(yPredictionTrain, yPredictionProbTrain) = self.predict(xTrain)
                #(yPredictionTest, yPredictionProbTest) = self.predict(xTest)
                #print("%s, %s, %s" % (str(i), str(self.loss(yTrain, yPredictionProbTrain)[0]), str(self.loss(yTest, yPredictionProbTest)[0])))
        #(yPredictionTrain, yPredictionProbTrain) = self.predict(xTrain)
        #(yPredictionTest, yPredictionProbTest) = self.predict(xTest)
        #print("%s, %s, %s" % (str(200), str(self.loss(yTrain, yPredictionProbTrain)[0]), str(self.loss(yTest, yPredictionProbTest)[0])))

    def predict(self, input, threshold=.5):
        yPredictionProb = []
        yPrediction = []
        for sample in input:
            value = self.forwardForPredict(sample)
            yPredictionProb.append(value)
            if value > threshold:
                yPrediction.append(1)
            else:
                yPrediction.append(0)

        return (yPrediction, yPredictionProb)

    def forwardForPredict(self, X):
        self.z = numpy.dot(X, self.weights[0]) # dot product of X (input) and first set of 3x2 weights
        self.z2 = sigmoid(self.z) # activation function
        self.z3 = numpy.dot(self.z2, self.weights[1]) # dot product of hidden layer (z2) and second set of 3x1 weights
        o = sigmoid(self.z3) # final activation function

        # if theres another hidden layer
        if len(self.weights) == 3:
            self.z4 = numpy.dot(o, self.weights[2])
            return sigmoid(self.z4)
        else:
            return o

    def loss(self, y, yPredicted):
        squares = []
        for i in range(len(y)):
            if (yPredicted[i] - y[i] <= 0):
                squares.append(0)
            else:                
                value = numpy.sqrt(yPredicted[i] - y[i])
                squares.append(value)
        return ((1 / 2) * sum(squares)) / len(y)
