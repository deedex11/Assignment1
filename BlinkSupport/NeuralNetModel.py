import collections
import numpy as np

class NeuralNetModel(object):

    def __init__(self):
        self.weights = [] # this is going ot be array of array of weights for each layer in order
        pass

    def fit(self, xTrain, yTrain, stepSize=0.1, numHiddenLayer=1, numNodePerHiddenLayer=2, numIteration=1, presetWeights=None):
        if presetWeights:
            self.weights = presetWeights

        print("Lngth of xtrain: " + str(len(xTrain)))
        print("this is feature length: " + str(len(xTrain[0])))
        #weights
        self.W1 = np.random.randn(numNodePerHiddenLayer, len(xTrain[0])) # (3x2) weight matrix from input to hidden layer
        print("Weight 1: " + str(self.W1))
        if (numHiddenLayer == 2):
            print("in the if")
            self.W2 = np.random.randn(numNodePerHiddenLayer, numNodePerHiddenLayer)
            self.W3 = np.random.randn(numNodePerHiddenLayer, 1) #
            print("Weight2: " + str(self.W2))
            print("Weight3: " + str(self.W3))
        else:
            self.W2 = np.random.randn(numNodePerHiddenLayer, 1) # (3x1) weight matrix from hidden to output layer
            print("Weight2" + str(self.W2))

        #For each iteration:
        for i in range(numIteration):
        #For each training sample:
        #Pass the sample through the network to get activations: foward propagation
        # Pass the training set through our neural network: #Forward Propagation
            hiddenLayer1 = self.sigmoid(np.dot(xTrain.T, self.W1.T))
            print("hidden layer 1: " + stR(hiddenLayer2))
            if self.W3:
                hiddenLayer2 = self.sigmoid(np.dot(outputLayer1, self.W2))
                
            outputLayer1 = self.sigmoid(np.dot(outputLayer1, self.weightsOutput))
             #Backpropagation
            # Calculate the error for layer 2 (The difference between the desired output
            # and the predicted output).  training_set_inputs, training_set_outputs, number_of_training_iterations
            if self.W3:
                layer3Change = (yTrain - hiddenLayer2) * self.sigmoid_derivative(outputlayer1)

            # Calculate the error for layer 2 (The difference between the desired output
            # and the predicted output).  training_set_inputs, training_set_outputs, number_of_training_iterations
            layer2Change = (yTrain - hiddenLayer2) * self.sigmoid_derivative(hiddenlayer2)
            print("leyer 2 change: " + str(layer2Change))

            # Calculate the error for layer 1 (By looking at the weights in layer 1,
            # we can determine by how much layer 1 contributed to the error in layer 2).
            layer1Change = layer2Change.dot(self.weightsOutput.T) * self.sigmoid_derivative(hiddenLayer1)

            # Adjusting weights
            self.W1 += xTrain.T.dot(layer1Change)
            self.W2 += outputLayer1.T.dot(layer2Change)
            print("self weight adjust: " + str(self.W1))

            if self.W3:
                self.W3 += outputLayer1.T.dot(layer3Change)

        # now 

    def predict(self, x):
        return [self.prediction for example in x]

    def loss(self, y, yPredicted):
        squares = []
        for i in range(len(y)):
            squares.append(sqrt(yPredicted[i] - y[i]))
        return (1 / len(y)) * sum(squares)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)