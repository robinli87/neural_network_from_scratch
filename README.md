# neural_network_from_scratch
A customisable, general purpose neural network build from very basic Python commands; no third party library is used. No numpy, no Keras, no Pytorch, no Tensorflow...  Only maths and random. Can multiprocess. 

On the input side, the network ingests an input dyad where each column is a datapoint (feature vector for one datapoint). The "run" method computes some "predicted " outputs according to your inputs and the parameters in the network. 
To train, use the "train" method. It compares the network's predictions to your provided "correct" output then carries out backpropagation to adjust the weights, so that eventually the "run" method can output exactly your "correct" outputs. The user must define the shape of the network: the number of nodes in the input layer, the number of nodes in the output layer, the number of hidden layers and the number of nodes in each hidden layer, using the argument "structure" in __init__ method of the NN class. This gives the user freedom to design the network to be any suitable shape and experiment with different architectures. 

There are adjustable hyperparameters such as tolerance, learning rates, parameters associated with activation functions etc... 

Multiprocessing is implemented to take full advantage of your CPU
Maximum number of simultaneous process = total number of nodes in the network. 

To run the program, execute fastmain.py

python3 fastmain.py
