import sys
import matplotlib
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs   #remebering input values
    def backward(self, dvalues): #backward pass(backpropogation)
        #parameters gradients
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        #values gradients
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs   #remebering input values
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues): #backward pass(backpropogation)
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0



class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))  #get the unnormalized probs
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)  #normalize them sample-wise
        self.output = probabilities
    def backward(self, dvalues):   #backward pass(backpropogation)
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        #clip the data to prevent division by 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    def backward(self, dvalues, y_true):    #backward pass(backpropogation)
        samples = len(dvalues)   #number of samples
        labels = len(dvalues[0])  #number of labels
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true/dvalues  #gradient calculation
        self.dinputs = self.dinputs/samples    #gradient normalization


class Softmax_Activation_CategoricalCrossEntropy_Loss_Combined():
    #create softmax and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()
    #forward pass
    def forward(self, inputs, y_true):
        self.activation.forward(inputs) #output lvl activation
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true) #return the calculated loss
    #backward pass(backpropogation)
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1   #calculate gradient
        self.dinputs = self.dinputs / samples    #normalize it


class SGD_Optimizer:
    def __init__(self, learning_rate=0.95, decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):  #if there is yet no momentum arrays, then create them
                layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            #weight updates
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            #bias updates
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.learning_rate * layer.dweights
            bias_updates = -self.learning_rate * layer.dbiases
        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1


X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 64)   #inputs are just xy data in this case so the first parameter must be 2
activation1 = Activation_ReLU()

dense2 = Layer_Dense(64, 3)   #there is 3 outputs on the previous level so the number of inputs on the next one is 3 aswell
loss_activation = Softmax_Activation_CategoricalCrossEntropy_Loss_Combined()

optimizer = SGD_Optimizer(decay=1e-3, momentum=0.96)

for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)

    #calculate the accuracy from output of loss_activation
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)
    if not epoch % 100:
        print(f'epoch: {epoch}, ' + f'acc: {accuracy:.3f}, ' + f'loss: {loss:.3f}, ' + f'lr: {optimizer.current_learning_rate}')

    #backpropogation
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

'''#gradients
print('dense1 dweights: ')
print(dense1.dweights)
print('dense1 dbiases: ')
print(dense1.dbiases)
print('dense2 dweights: ')
print(dense2.dweights)
print('dense2 dbiases: ')
print(dense2.dbiases)
'''


'''
softmax_outputs = np.array([[0.3, 0.15, 0.3],
                            [0.1, 0.6, 0.7],
                            [0.2, 0.85, 0.3]])

class_targets = np.array([0, 1, 1])

softmax_loss = Softmax_Activation_CategoricalCrossEntropy_Loss_Combined()
softmax_loss.backward(softmax_outputs, class_targets)
dvalues1 = softmax_loss.dinputs

activation = Activation_Softmax()
activation.output = softmax_outputs
loss = Loss_CategoricalCrossEntropy()
loss.backward(softmax_outputs, class_targets)
activation.backward(loss.dinputs)
dvalues2 = activation.dinputs

print('Separate:')
print(dvalues2)
print('Combined:')
print(dvalues1)

------------------------------------------------------------------------- 
 
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]

layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer2_outputs)



output = [inputs[0] * weights1[0] + inputs[1] * weights1[1] + inputs[2] * weights1[2] + inputs[3] * weights1[3] + bias1,
          inputs[0] * weights2[0] + inputs[1] * weights2[1] + inputs[2] * weights2[2] + inputs[3] * weights2[3] + bias2,
          inputs[0] * weights3[0] + inputs[1] * weights3[1] + inputs[2] * weights3[2] + inputs[3] * weights3[3] + bias3]
print(output)
'''
