def fullyConnected1D(numInputNeurons, numOutputNeurons, inputActivations, weights2D):
    outputActivations = []
    for j in range(numOutputNeurons):
        output = 0
        for i in range(numInputNeurons):
            output += inputActivations[i] * weights2D[i][j]
        outputActivations.append(output)
    return outputActivations
    
# While
def convolution1D(numInputNeurons, inputActivations, filterSize, stride, weights):
    outputActivations = []
    i = 0
    start_idx = i * stride
    while (start_idx + filterSize - 1) < numInputNeurons:
        output = 0
        for j in range(filterSize):
            output += inputActivations[start_idx + j] * weights[j]
        outputActivations.append(output)
        i += 1
    return outputActivations


# For
def convolution1D(numInputNeurons, inputActivations, filterSize, stride, weights):
    outputActivations = []
    for i in range(((numInputNeurons - filterSize + 1) + 1) // stride): 
        output = 0
        start_inx = i * stride
        if start_inx + filterSize - 1 >= numInputNeurons:
            raise IndexError('Out-of-bounds')
        for j in range(filterSize):
            output += inputActivations[start_inx + j] * weights[j]
        outputActivations.append(output)
    return outputActivations