import os
import argparse
import tensorflow as tf

from shutil import rmtree
from neural_network import NeuralNetwork

def main():
    parser = argparse.ArgumentParser(description='parser')
    parser.add_argument('-t', help='teach the network')
    parser.add_argument('-e', type=int, help='amount of epochs', default=1000)
    parser.add_argument('-b', type=int, help='batch_size', default=5)
    args = parser.parse_args()

    epochs = args.e
    batch_size = args.b

    if args.t:
        nn = NeuralNetwork(epochs, batch_size)
        nn.run()
        return


if __name__ == "__main__":
    main()
