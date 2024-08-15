"""
Liquid Neural Network
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

def initialize_weights(input_dim, reservoir_dim, output_dim, spectral_radius):
    """
    This function initializes the weight matrices of our LNN and returns them

    Parameters
    ----------
    input_dim :
    reservoir_dim :
    output_dim :
    spectral_radius :

    Returns
    -------

    """
    # initialize reservoir weights randomly

    reservoir_weights = np.random.randn(reservoir_dim, reservoir_dim)

    # scale reservor weights to acheive desired spectral radium

    reservoir_weights *= spectral_radius / np.max(np.abs(np.linalg.eigvals(reservoir_weights)))

    # initialize input-to-reservoir weights randomly
    input_weights = np.random.randn(reservoir_dim, input_dim) # why the double parentheses??

    # initialize output weights to zero
    output_weights = np.zeros((reservoir_dim, output_dim))

    return reservoir_weights, input_weights, output_weights

def train_lnn(input_data, labels, reservoir_weights, input_weights, output_weights, leak_rate, num_epochs):

    num_samples = input_data.shape[0]
    reservoir_dim = reservoir_weights.shape[0]
    reservoir_states = np.zeros((num_samples, reservoir_dim))

    for epoch in range (num_epochs):
        for i in range(num_samples):
            if i > 0:
                reservoir_states[i, :] = (1-leak_rate) * reservoir_states[i - 1, :]
            reservoir_states[i, :] += leak_rate * np.tanh(np.dot(input_weights, input_data[i, :]) + np.dot(reservoir_weights, reservoir_states[i, :]))

        # train output weights
        output_weights = np.dot(np.linalg.pinv(reservoir_states), labels)

        # compute training accuracy

        train_predictions = np.dot(reservoir_states, output_weights)
        train_accuracy = np.mean(np.argmax(train_predictions, axis=1) == np.argmax(labels, axis=1))
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Accuracy: {train_accuracy:.4f}")

    return output_weights

def predict_lnn(input_data, reservoir_weights, input_weights, output_weights, leak_rate):

    num_samples = input_data.shape[0]
    reservoir_dim = reservoir_weights.shape[0]
    reservoir_states = np.zeros((num_samples, reservoir_dim))

    for i in range(num_samples):
        #update res state
        if i >0:
            reservoir_states[i, :] = (1 - leak_rate) * reservoir_states[i - 1, :]
        reservoir_states[i, :] += leak_rate * np.tanh(np.dot(input_weights, input_data[i, :]) + np.dot(reservoir_weights, reservoir_states[i, :]))

    # compute predicitons using output weights
    predictions = np.dot(reservoir_states, output_weights)
    return predictions


# load data and pre-process
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
x_train = x_train.reshape((60000, 784)) / 255.0
x_test = x_test.reshape((10000, 784)) / 255.0

# set LNN hyperparameters
input_dim = 784
reservoir_dim = 1000
output_dim = 10
leak_rate = 0.75  # default to 0.1 best results at 1000 res_dim 0.5
spectral_radius = 0.9
num_epochs = 10

# initialize weights
reservoir_weights, input_weights, output_weights = initialize_weights(input_dim, reservoir_dim, output_dim, spectral_radius)

output_weights = train_lnn(x_train, y_train, reservoir_weights, input_weights, output_weights, leak_rate, num_epochs)

# eval on test set
test_predictions = predict_lnn(x_test, reservoir_weights, input_weights, output_weights, leak_rate)

test_accuracy = np.mean(np.argmax(test_predictions, axis=1)== np.argmax(y_test, axis=1))
print(f"Test Accuracy: {test_accuracy:.4f}")