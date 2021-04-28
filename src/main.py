import gzip
import json
import os
import pickle
import numpy
from PIL import Image
import drawing
import network

"""
This project builds off of the neural network template provided by Michael
Nielson in his online book: http://neuralnetworksanddeeplearning.com
"""


def load_data():
    """
    Both entries in the tuple are arrays. The first array contains a number
    of 28x28 ndarrays, representing image pixels. The second array contains
    the corresponding digits in those images.

    :return: MNIST data as a tuple of the tuples described above
    """
    with gzip.open('../data/mnist.pkl.gz', 'rb') as f:
        training_data, validation_data, test_data = pickle.load(
            f, encoding='latin1')
        return training_data, validation_data, test_data


def increase_contrast(data):
    """
    Separates data more clearly by setting colour to either black or white

    :param data: Tuple of arrays representing MNIST data
    :return: MNIST data with colour values of only 0 or 1
    """
    threshold = 0.3
    for x in range(len(data[0])):
        if data[0][x][0] > threshold:
            data[0][x][0] = 1.0
        else:
            data[0][x][0] = 0.0

    return data


def format_data():
    """
    Formats data for ease of further calculations

    :return: Tuple of arrays representing MNIST data
    """
    raw_training, raw_validation, raw_test = load_data()

    training_inputs = [numpy.reshape(x, (IMG_PIXELS, 1))
                       for x in raw_training[0]]
    training_inputs = increase_contrast(training_inputs)
    training_results = [vectorized_result(y) for y in raw_training[1]]
    training_data = list(zip(training_inputs, training_results))

    validation_inputs = [numpy.reshape(x, (IMG_PIXELS, 1))
                         for x in raw_validation[0]]
    validation_inputs = increase_contrast(validation_inputs)
    validation_data = list(zip(validation_inputs, raw_validation[1]))

    test_inputs = [numpy.reshape(x, (IMG_PIXELS, 1)) for x in raw_test[0]]
    test_inputs = increase_contrast(test_inputs)
    test_data = list(zip(test_inputs, raw_test[1]))

    return training_data, validation_data, test_data


def vectorized_result(digit):
    """
    :param digit: number from 0 to 9
    :return: array representation of 10-d vector
    """
    arr = numpy.zeros((10, 1))
    arr[digit] = 1.0

    return arr


def train_network(neuron_list, epochs, mini_batch_size, learn_rate,
                  reg_lambda):
    """
    Initializes a neural network and begins the SGD process

    :param neuron_list: list with number of neurons per layer
    :param epochs: number of training rounds
    :param mini_batch_size: number of images to use per epoch
    :param learn_rate: float that adjusts how fast the network converges
    :param reg_lambda: float that determines network's self-regularization
    :return: Network object representing the neural network
    """
    training_data, validation_data, test_data = format_data()
    neural_network = network.Network(neuron_list)
    neural_network.stochastic_gradient_descent(
        training_data, validation_data, epochs, mini_batch_size, learn_rate,
        reg_lambda, TRAINING_DATA_SIZE, test_data=test_data)

    return neural_network


def load(filename):
    """
    Loads in a previously saved neural network file (JSON)

    :param filename:
    :return: Network object
    """
    with open(filename, "r") as f:
        data = json.load(f)

        copy = network.Network(data["sizes"])
        copy.weights = [numpy.array(w) for w in data["weights"]]
        copy.biases = [numpy.array(b) for b in data["biases"]]
        print("Load successful")

        return copy


# Tests a given image
def test_image(img, adjusted_black_value, invert):
    """
    Tests a given image on the current neural network

    :param img: PIL Image object
    :param adjusted_black_value: 0-1 decimal representing the colour black
    :param invert: bool for whether or not the image is to be colour-inverted
    """
    img_length = int(IMG_PIXELS ** 0.5)
    img = img.resize((img_length, img_length), Image.ANTIALIAS)
    img_arr = numpy.array(img, dtype="float32")
    img_arr = (img_arr[:, :, 0].flatten()) / 255

    img_arr[img_arr < 0.5] = 0.0
    img_arr[img_arr > 0.4] = adjusted_black_value

    if invert:
        img_arr = [1.0 - x for x in img_arr]

    formatted_arr = numpy.reshape(img_arr, (IMG_PIXELS, 1))
    user_net.test(formatted_arr)


def print_options():
    """Prints all user input options"""
    print("Choose an action below by entering the corresponding number:")
    print("0) Quit")
    print("1) Train new neural network using the MNIST database")
    print("2) Load saved neural network")
    print("3) Submit picture of digit to test using neural network")
    print("4) Draw your digit right now")
    print("5) Save current neural network")


def build_net():
    """
    Gets user input to build neural net

    :return: trained Network object
    """
    network_list = [IMG_PIXELS];

    print("Enter the number of hidden layers you would like, the numbers of "
          "\nneurons for those hidden layers, the number of training epochs,"
          "\nthe mini batch size, the learn rate, and the lambda of "
          "\nregularization one by one (enter ? if you do not understand "
          "\nthese terms).\n\nNumber of hidden layers:")
    hidden_layers = (input())

    while hidden_layers == '?':
        print("The number of neurons in each hidden layer and the number of"
              "\nhidden layers both determine how complex a neural network"
              "\nis. A training epoch describes when a network learns from all"
              "\nits training data exactly once. The mini batch size"
              "\ndetermines the number of data points taken into account"
              "\nbefore each update of the weights and biases. The learn rate"
              "\nis a measure of how aggressively the network tries to correct"
              "\nitself, and the lambda of regularization determines how hard"
              "\nthe network tries to keep its weights at smaller values. If"
              "\nyou don't know what values to put, try inputting 1, 50, 20,"
              "\n 10, 0.5, 2 as a starting point."
              "\n\nNumber of hidden layers:")
        hidden_layers = (input())
    hidden_layers = int(hidden_layers)

    for x in range(hidden_layers):
        print("Neurons in hidden layer number " + str(x + 1) + ":")
        network_list.append(int(input()));
    network_list.append(10);

    print("Training epochs:")
    training_epochs = int(input())

    print("Mini-batch size:")
    mini_batch_size = int(input())

    print("Learn rate:")
    learn_rate = float(input())

    print("Lambda of regularization:")
    regularization_lambda = float(input())
    return train_network(network_list, training_epochs, mini_batch_size,
                         learn_rate, regularization_lambda)


def upload_test_image():
    """If user chooses to upload image"""
    print("Draw a digit using black marker on white paper and take a square"
          "\npicture. For better results, center the digit in the image.")
    file_path = input("Input the file path to the saved jpg image:")
    image = Image.open(file_path)
    test_image(image, 0.941, True)


def draw_test_image():
    """If user chooses to draw image"""
    print("For the best results, draw the digits in the middle of the box.")
    drawing.Drawing(300, 100, int(IMG_PIXELS ** 0.5))
    image = Image.open(os.getcwd().rsplit('\\', 1)[0] + "/drawIn/input.jpg")
    test_image(image, 1.00, False)


if __name__ == '__main__':
    IMG_PIXELS = 784
    TRAINING_DATA_SIZE = 50000

    user_net = network.Network([IMG_PIXELS, 1, 10])
    action = -1

    print("Hello! This program will demonstrate the ability of a neural net in"
          "learning to recognize handwritten digits.")

    while action != 0:
        print_options()
        action = input()

        if action == '0':
            break
        elif action == '1':
            user_net = build_net()
        elif action == '2':
            path = input("Please enter the valid file path that you wish to"
                         " load your neural net from:")
            user_net = load(path)
        elif action == '3':
            upload_test_image()
        elif action == '4':
            draw_test_image()
        elif action == '5':
            path = input("Please enter the valid file path that you wish to"
                         "save the neural net as:")
            user_net.save(path)
        else:
            print("None of the above were entered. Please try again.")

        print("")
