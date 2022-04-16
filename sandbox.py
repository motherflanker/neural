#everything bellow is the models manual instantiations before i got the model class
'''
X, y = spiral_data(samples=1000, classes=2)

y = y.reshape(-1, 1)

dense1 = Layer_Dense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)   #inputs are just xy data in this case so the first parameter must be 2

activation1 = Activation_ReLU()

dense2 = Layer_Dense(64, 1)   #there is 3 outputs on the previous level so the number of inputs on the next one is 3 aswell

activation2 = Activation_Sigmoid()

loss_function = Loss_BinaryCrossEntropy()

#loss_activation = Softmax_Activation_CategoricalCrossEntropy_Loss_Combined()

#optimizer = SGD_Optimizer(decay=1e-3, momentum=0.96)
#optimizer = AdaGrad_Optimizer(decay=1e-4)
#optimizer = RMSprop_Optimizer(decay=1e-4)


X, y = sine_data()
dense1 = Layer_Dense(1, 64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 64)
activation2 = Activation_ReLU()
dense3 = Layer_Dense(64, 1)
activation3 = Activation_Linear()
loss_function = Loss_MeanSquaredError()
optimizer = Adam_Optimizer(learning_rate=0.005, decay=1e-3)
accuracy_precision = np.std(y) / 250

for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)
    data_loss = loss_function.calculate(activation3.output, y)


    regularization_loss = \
    loss_function.regularization_loss(dense1) + \
    loss_function.regularization_loss(dense2) + \
    loss_function.regularization_loss(dense3)
    loss = data_loss + regularization_loss


    #nonbinary

    data_loss = loss_activation.forward(dense2.output, y)

    regularization_loss = \
        loss_function.regularization_loss(dense1) + \
        loss_function.regularization_loss(dense2)
    loss = data_loss + regularization_loss

    #binary
    regularization_loss = \
        loss_activation.loss.regularization_loss(dense1) + \
        loss_activation.loss.regularization_loss(dense2)
    loss = data_loss + regularization_loss

    #calculate the accuracy from output of loss_activation
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    #calculate accuracy from output of activation2 || binary
    predictions = (activation2.output > 0.5) * 1  #binary array in the brackets returns true\false values
    accuracy = np.mean(predictions == y)          # *1 => array of 1\0


    predictions = activation3.output
    accuracy = np.mean(np.absolute(predictions - y) < accuracy_precision)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}, (' +
              f'dataLoss: {data_loss:.3f}, ' +
              f'regLoss: {regularization_loss:.3f}), ' +
              f'lr: {optimizer.current_learning_rate}')
    loss_function.backward(activation3.output, y)
    activation3.backward(loss_function.dinputs)
    dense3.backward(activation3.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()


    # binary same for regression
    loss_function.backward(activation2.output, y)
    activation2.backward(loss_function.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    # Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()


    #backpropogation nonbinary
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()
    #regression test
    import matplotlib.pyplot as plt

    X_test, y_test = sine_data()

    dense1.forward(X_test)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)

    plt.plot(X_test, y_test)
    plt.plot(X_test, activation3.output)




    # TESTING THE MODEL || binary
    X_test, y_test = spiral_data(samples=100, classes=2)
    y_test = y_test.reshape(-1, 1)
    dense1.forward(X_test)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    loss = loss_function.calculate(activation2.output, y_test)
    predictions = (activation2.output > 0.5) * 1
    accuracy = np.mean(predictions == y_test)
    print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')


    # TESTING THE MODEL || nonbinary
    X_test, y_test = spiral_data(samples=100, classes=3)
    dense1.forward(X_test)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y_test)
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y_test.shape) == 2:
        y_test = np.argmax(y_test, axis=1)
    accuracy = np.mean(predictions == y_test)
    print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')


#gradients
print('dense1 dweights: ')
print(dense1.dweights)
print('dense1 dbiases: ')
print(dense1.dbiases)
print('dense2 dweights: ')
print(dense2.dweights)
print('dense2 dbiases: ')
print(dense2.dbiases)



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
'''
