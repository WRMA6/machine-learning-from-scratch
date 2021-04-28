# Python Machine Learning Program
This program is made using Python and implements a neural network to classify handwritten digits.

The following features are in the program:
- Creating and training a neural network using the MNIST datset
- Testing the neural network either by using a pre-existing jpg image or by drawing on a tkinter canvas
- Loading and saving neural networks from and to a JSON file

Notable libraries used:
numpy, tkinter, PIL, json

There should not be too much setup needed to run the code. It was written for Python 3 on the PyCharm IDE.

Most neural network concepts were learned from Michael Nielson's online book at  http://neuralnetworksanddeeplearning.com/chap1.html. The neural network was built off of the book's suggested structure.

In further detail, the neural network uses stochastic graident descent to train. There are 784 input neurons (one for each pixel of the input image) and 10 output neurons (for the 10 possible digits). The program automatically uses the 50000 samples pre-downloaded from the MNIST database to do so. In order to learn (ie. change its weights and biases), the standard approach of backpropagation is used. All the neurons work off of the sigmoid activation function, so a cross-entropy cost function was used to ensure that learning would not stagnate if the network still had a lot of it to do. L2 regularization was implemented to keep the weights at a healthy size. Finally, the program automatically stops training if it detects that learning has saturated from the test results to prevent overfitting to the training data.

For the input image option, all images are automatically turned into black and white pixels only, then shrunk to 28x28 pictures and inverted so that it looks like white writing on a black background. This allows normal writing on paper to be used as input.

Below, you can see a demo of the code at work, where a digit is drawn for an already trained network. The program is able to correctly identify the digit drawn as a 3.

![](/MLProjectDemo.gif?raw=true "Screencap")