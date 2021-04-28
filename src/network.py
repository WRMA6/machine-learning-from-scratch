import json
import random
import numpy


class Network(object):
    """
    Represents a neural network with biases and weights and training methods
    """

    def __init__(self, sizes):
        """
        Initializes all neurons in the network
        The weights are initialized to 1/sqrt(# of input neurons)

        :param sizes: List of integers representing neurons in each layer
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [numpy.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [(1 / (x ** 0.5)) * numpy.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feed_forward(self, input_vector):
        """
        Calculates the vector output for a vector input to the network

        :param input_vector: vector representing the previous layer's output
        :return: vector output representing current layer's output
        """
        output_vector = input_vector
        for bias, weight in zip(self.biases, self.weights):
            output_vector = sigmoid(numpy.dot(weight, output_vector) + bias)
        return output_vector

    def stochastic_gradient_descent(
            self, training_data, validation_data, epochs, mini_batch_size,
            learn_rate, reg_lambda, training_size, test_data=None):
        """
        Splits data into mini-batches and then calls on update_mini_batch
        The network is tested after each epoch if test data is given

        :param training_data: tuple of lists representing images of digits
        :param validation_data: tuple of lists representing images of digits
        :param epochs: number of training rounds
        :param mini_batch_size: number of images to use per epoch
        :param learn_rate: float that adjusts how fast the network converges
        :param reg_lambda: float that determines network's self-regularization
        :param training_size: number of data samples available to train with
        :param test_data: tuple of lists representing images of digits
        """
        highest = 0
        sat_streak = 0
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size]
                            for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learn_rate, reg_lambda,
                                       training_size)
            proceed, highest, sat_streak = self.validate(validation_data,
                                                         highest, sat_streak)
            if not proceed:
                print("Learning has been saturated in training epoch {0}. "
                      "Final set gives {1} / {2} correct"
                      .format(j + 1, self.test_batch(test_data), n_test))
                break
            else:
                if test_data:
                    print("Finished training epoch {0}: {1} / {2} correct"
                          .format(j + 1, self.test_batch(test_data), n_test))
                else:
                    print("Epoch {0} complete, "
                          "no test data to compare against".format(j))

    def update_mini_batch(self, mini_batch, learn_rate, reg_lambda,
                          training_size):
        """
        Calculates rate of change of cost function relative to each weight and
        bias by calling backpropagation function
        Each weight and bias is updated so that cost function changes by
        -(learning rate)*(gradient)^2 (always decreasing towards 0)

        :param mini_batch: tuple of lists representing images of digits
        :param learn_rate: float that adjusts how fast the network converges
        :param reg_lambda: float that determines network's self-regularization
        :param training_size: number of data samples available to train with
        :return:
        """
        grad_b = [numpy.zeros(bias.shape) for bias in self.biases]
        grad_w = [numpy.zeros(weight.shape) for weight in self.weights]

        for pixels, answer in mini_batch:
            change_b, change_w = self.backpropagation(pixels, answer)
            grad_b = [bias + change for bias, change in zip(grad_b, change_b)]
            grad_w = [weight + change for weight, change
                      in zip(grad_w, change_w)]

        self.weights = [
            weight * (1 - (learn_rate * reg_lambda) / training_size)
            - (learn_rate / len(mini_batch)) * grad
            for weight, grad in zip(self.weights, grad_w)
        ]

        self.biases = [bias - (learn_rate / len(mini_batch)) * grad
                       for bias, grad in zip(self.biases, grad_b)]

    def backpropagation(self, pixels, answer):
        """
        Moves from output layer to input layer, using chain-rule to calculate
        all the required gradients

        :param pixels: array representing pixels of the image
        :param answer: the digit in the image
        :return: nparrays for cost function grad. w/r to each weight and bias
        """
        # Feed forward
        grad_b = [numpy.zeros(b.shape) for b in self.biases]
        grad_w = [numpy.zeros(w.shape) for w in self.weights]
        activation = pixels
        activations = [pixels]
        weighted_inputs = []

        for bias, weight in zip(self.biases, self.weights):
            weighted_input = numpy.dot(weight, activation) + bias
            weighted_inputs.append(weighted_input)
            activation = sigmoid(weighted_input)
            activations.append(activation)

        # Quadratic cost function results in this simple derivative
        grad_cost = activations[-1] - answer
        # Last layer cost derivatives
        grad_b[-1] = grad_cost
        grad_w[-1] = numpy.dot(grad_cost, activations[-2].transpose())

        for layer in range(2, self.num_layers):
            weighted_input = weighted_inputs[-layer]
            sp_input = sigmoid_prime(weighted_input)

            # Calculate grad. cost for layer x using grad. cost of layer x+1
            grad_cost = numpy.dot(
                self.weights[-layer + 1].transpose(), grad_cost) * sp_input

            grad_b[-layer] = grad_cost
            grad_w[-layer] = numpy.dot(grad_cost,
                                       activations[-layer - 1].transpose())

        return grad_b, grad_w

    def test(self, test_data):
        """
        Prints out what the net identified the data as

        :param test_data: tuple of lists representing images of digits
        """
        output = self.feed_forward(test_data)
        result = numpy.argmax(output)
        print("The program thinks your digit is a " + str(result) + ".")

    def test_batch(self, test_data):
        """
        :param test_data: tuple of lists representing images of digits
        :return: Number of test cases that were correct
        """
        test_results = [(numpy.argmax(self.feed_forward(pixels)), answer)
                        for (pixels, answer) in test_data]
        return sum(int(result == answer) for (result, answer) in test_results)

    def validate(self, validation_data, highest, sat_streak):
        """
        Marks the neural net's performance, then looks at recent result to
        decide if learning has saturated

        :param validation_data: tuple of lists representing images of digits
        :param highest: highest recent score
        :param sat_streak: number of epochs since score last improved
        :return: tuple with updated info on current training saturation
        """
        if sat_streak >= 5:
            result = self.test_batch(validation_data)
            if result > highest - 10:
                return False, highest, sat_streak
        else:
            result = self.test_batch(validation_data)
            if result >= highest:
                highest = result
                sat_streak = 0
            else:
                sat_streak += 1
        return True, highest, sat_streak

    def save(self, filename):
        """
        Saves this neural net to a file as JSON

        :param filename: string with file path to save net to
        """
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases]}
        with open(filename, "w") as f:
            json.dump(data, f)


def sigmoid(x):
    """
    Sigmoid function is the activation function for nodes in the net

    :param x: nparray
    :return: sigmoid(x)
    """
    return 1.0 / (1.0 + numpy.exp(-x))


def sigmoid_prime(x):
    """
    Derivative of sigmoid function (used in backpropagation)

    :param x: nparray
    :return: sigmoid'(x)
    """
    return sigmoid(x) * (1 - sigmoid(x))
