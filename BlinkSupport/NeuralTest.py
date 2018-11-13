import NeuralNetModel

xTrain = [[1, 0.5]]
yTrain = [1]

print('test1')
model = NeuralNetwModel()
presetWeights = [
    [
        # neuron 1 - hidden layer 1
        {'w0':0.5, 'weights':[-1.0, 1.0]},
        # neuron 2 - hidden layer 1
        {'w0':1.0, 'weights':[0.5, -1.0]},
    ],
    # output layer
    [
        {'w0':0.25, 'weights':[1.0, 1.0]},
    ]
]
model.fit(xTrain, yTrain, stepSize=0.1, numHiddenlayer=1, numNodePerHiddenLayer=2, numIteration=1, presetWeights=presetWeights)
model.print()

print('test2')
model = NeuralNetModel()
presetWeights = [
    # layer 1
    [
        # neuron 1 
        {'w0':0.5, 'weights':[-1.0, 1.0]},
        # neuron 2
        {'w0':1.0, 'weights':[0.5, -1.0]},
    ],
    # layer 2
    [
        # neuron 1
        {'w0':0.75, 'weights':[0.5, 0.5]},
        # neuron 2
        {'w0':0.25, 'weights':[1.0, -1.0]},
    ],
    # output layer
    [
        {'w0':0.25, 'weights':[1.0, 1.0]},
    ]
]
model.fit(xTrain, yTrain, stepSize=0.1, numHiddenlayer=2, numNodePerHiddenLayer=2, parallel=False, numIteration=1, presetWeights=presetWeights)
model.print()