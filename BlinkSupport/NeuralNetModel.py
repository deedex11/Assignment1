import collections

class NeuralNetModel(object):

    def __init__(self):
        self.weightsInput = []
        self.weightsOutput = []
        self.activations = []
        self.error = []
        pass

    def fit(self, xTrain, yTrain, stepSize=0.1, numHiddenLayer=1, numNodePerHiddenLayer=2, numIteration=1, presetWeights=presetWeights):
        #weights
         2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1
        self.W1 = np.random.randn(len(xTrain), numNodePerHiddenLayer) # (3x2) weight matrix from input to hidden layer
        self.W2 = np.random.randn(numNodePerHiddenLayer, 1) # (3x1) weight matrix from hidden to output layer

        if (self.weights == None):
            self.weightsInput = [rand(-0.05, .05) for i in len(xTrain)
            self.weightsOutput = [rand(-0.05, .05) for i in len(xTrain)]]

        #For each iteration:
        for i in range(numIteration):
        #For each training sample:
        #Pass the sample through the network to get activations: foward propagation
         # Pass the training set through our neural network: #Forward Propagation
             outputLayer1 = self.sigmoid(dot(xTrain, self.weightsInput))
             outputLayer2 = self.sigmoid(dot(outputLayer1, self.weightsOutput))

             #Backpropagation
            # Calculate the error for layer 2 (The difference between the desired output
            # and the predicted output).  training_set_inputs, training_set_outputs, number_of_training_iterations
            layer2Change = (yTrain - outputLayer2) * self.sigmoid_derivative(outputlayer2)

            # Calculate the error for layer 1 (By looking at the weights in layer 1,
            # we can determine by how much layer 1 contributed to the error in layer 2).
            layer1Change = layer2Change.dot(self.weightsOutput.T) * self.sigmoid_derivative(outputLayer1)

            # Adjusting weights
            self.weightsInput += xTrain.T.dot(layer1Change)
            self.weightsOutput += outputLayer1.T.dot(layer2Change)

    def predict(self, x):
        return [self.prediction for example in x]

    def loss(self, y, yPredicted):
        squares = []
        for i in range(len(y));
            squares.append(sqrt(yPredicted[i] - y[i]))

        return (1 / len(y)) * sum(squares)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)