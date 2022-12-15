from Linear import Linear
from CrossEntropy import CrossEntropy
from SoftMax import SoftMax

import numpy as np

import urllib.request
import os

import tarfile


def train_and_test(
    network,
    loss,
    train_data,
    train_labels,
    test_data,
    test_labels,
    batch_size=600,
    num_epochs=100,
    learning_rate=0.1,
):
    """
    Train the network on the MNIST dataset

    Args:
    network: Neural network
    loss: Loss function
    train_data: Training data
    train_labels: Training labels
    test_data: Test data
    test_labels: Test labels
    batch_size: Batch size
    num_epochs: Number of epochs
    learning_rate: Learning rate
    """
    # Number of iterations
    num_iterations = int(train_data.shape[0] / batch_size) * num_epochs

    # Train the network
    for i in range(num_iterations):
        # Select the mini-batch
        batch_start = (i * batch_size) % train_data.shape[0]
        batch_end = batch_start + batch_size
        batch_data = train_data[batch_start:batch_end]
        batch_labels = train_labels[batch_start:batch_end]

        for module in network:
            batch_data = module.fprop(batch_data)

        E = loss.fprop(batch_data, batch_labels)
        dz = loss.bprop(1 / batch_size)

        for module in reversed(network):
            dz = module.bprop(dz)
        for module in network:
            module.update(learning_rate)

        # Print the loss every 100 iterations
        if i % 100 == 0:
            print("Iteration: %d, Loss: %f" % (i, E))

    # Test the network
    num_correct = 0
    for i in range(test_data.shape[0]):
        # Forward propagation
        output = test_data[i]
        for module in network:
            output = module.fprop(output)

        # Compute the accuracy
        if np.argmax(output) == np.argmax(test_labels[i]):
            num_correct += 1

    print("Accuracy: %f" % (num_correct / test_data.shape[0]))
    print("Number of errors: %d" % (test_data.shape[0] - num_correct))


def test_2_d():
    url = "https://omnomnom.vision.rwth-aachen.de/data/mnist.tgz"
    filename = "mnist.tgz"
    if not os.path.exists(filename):
        print("Downloading MNIST dataset...")
        urllib.request.urlretrieve(url, filename)
        print("Download complete.")
    else:
        print("MNIST dataset already downloaded.")

    if not os.path.exists("mnist"):
        print("Extracting MNIST dataset...")
        tar = tarfile.open(filename)
        tar.extractall(path="mnist")
        tar.close()
        print("Extraction complete.")
    else:
        print("MNIST dataset already extracted.")

    # files are mnist-train-data.csv, mnist-train-labels.csv, mnist-test-data.csv, mnist-test-labels.csv
    train_data = np.loadtxt("mnist/mnist-train-data.csv")
    train_labels = np.loadtxt("mnist/mnist-train-labels.csv")
    test_data = np.loadtxt("mnist/mnist-test-data.csv")
    test_labels = np.loadtxt("mnist/mnist-test-labels.csv")
    print("MNIST dataset loaded.")

    # Normalize the data
    train_data = train_data / 255.0
    test_data = test_data / 255.0

    # One-hot encode the labels
    train_labels_one_hot = np.zeros((train_labels.shape[0], 10))
    train_labels_one_hot[np.arange(train_labels.shape[0]), train_labels.astype(int)] = 1
    test_labels_one_hot = np.zeros((test_labels.shape[0], 10))
    test_labels_one_hot[np.arange(test_labels.shape[0]), test_labels.astype(int)] = 1

    # Mini-batch size
    batch_size = 600

    # Number of epochs
    num_epochs = 100

    # Learning rate
    learning_rate = 0.1

    # Initialize the modules
    # using the following network architecture: Linear(28 ∗ 28, 10), SoftMax and the CrossEntropy criterion
    linear = Linear(28 * 28, 10)
    softmax = SoftMax()
    network = [linear, softmax]
    loss = CrossEntropy()

    train_and_test(
        network,
        loss,
        train_data,
        train_labels_one_hot,
        test_data,
        test_labels_one_hot,
        batch_size,
        num_epochs,
        learning_rate,
    )


def test_2_e_1():
    url = "https://omnomnom.vision.rwth-aachen.de/data/cifar10.tgz"
    filename = "cifar10.tgz"
    if not os.path.exists(filename):
        print("Downloading CIFAR-10 dataset...")
        urllib.request.urlretrieve(url, filename)
        print("Download complete.")
    else:
        print("CIFAR-10 dataset already downloaded.")

    if not os.path.exists("cifar10"):
        print("Extracting CIFAR-10 dataset...")
        tar = tarfile.open(filename)
        tar.extractall(path="cifar10")
        tar.close()
        print("Extraction complete.")
    else:
        print("CIFAR-10 dataset already extracted.")

    # files are cifar10-train-data.csv, cifar10-train-labels.csv, cifar10-test-data.csv, cifar10-test-labels.csv
    train_data = np.loadtxt("cifar10/cifar10-train-data.csv")
    train_labels = np.loadtxt("cifar10/cifar10-train-labels.csv")
    test_data = np.loadtxt("cifar10/cifar10-test-data.csv")
    test_labels = np.loadtxt("cifar10/cifar10-test-labels.csv")
    print("CIFAR-10 dataset loaded.")

    # Normalize the data
    train_data = train_data / 255.0
    test_data = test_data / 255.0

    # One-hot encode the labels
    train_labels_one_hot = np.zeros((train_labels.shape[0], 10))
    train_labels_one_hot[np.arange(train_labels.shape[0]), train_labels.astype(int)] = 1
    test_labels_one_hot = np.zeros((test_labels.shape[0], 10))
    test_labels_one_hot[np.arange(test_labels.shape[0]), test_labels.astype(int)] = 1

    # Mini-batch size
    batch_size = 600

    # Number of epochs
    num_epochs = 100

    # Learning rate
    learning_rate = 0.1

    # Initialize the modules
    # using the following network architecture: Linear(32 ∗ 32 ∗ 3, 10), SoftMax and the CrossEntropy criterion
    linear = Linear(32 * 32 * 3, 10)
    softmax = SoftMax()
    network = [linear, softmax]
    loss = CrossEntropy()

    train_and_test(
        network,
        loss,
        train_data,
        train_labels_one_hot,
        test_data,
        test_labels_one_hot,
        batch_size,
        num_epochs,
        learning_rate,
    )


def test_2_e_2(use_coarse_labels=False):
    url = "https://omnomnom.vision.rwth-aachen.de/data/cifar100.tgz"
    filename = "cifar100.tgz"
    if not os.path.exists(filename):
        print("Downloading CIFAR-100 dataset...")
        urllib.request.urlretrieve(url, filename)
        print("Download complete.")
    else:
        print("CIFAR-100 dataset already downloaded.")

    if not os.path.exists("cifar100"):
        print("Extracting CIFAR-100 dataset...")
        tar = tarfile.open(filename)
        tar.extractall(path="cifar100")
        tar.close()
        print("Extraction complete.")
    else:
        print("CIFAR-100 dataset already extracted.")

    # files are cifar100-train-data.csv, cifar100-train-labels.csv, cifar100-test-data.csv, cifar100-test-labels.csv
    train_data = np.loadtxt("cifar100/cifar100-train-data.csv")
    if use_coarse_labels:
        train_labels = np.loadtxt("cifar100/cifar100-train-coarse-labels.csv")
    else:
        train_labels = np.loadtxt("cifar100/cifar100-train-fine-labels.csv")
    test_data = np.loadtxt("cifar100/cifar100-test-data.csv")
    if use_coarse_labels:
        test_labels = np.loadtxt("cifar100/cifar100-test-coarse-labels.csv")
    else:
        test_labels = np.loadtxt("cifar100/cifar100-test-fine-labels.csv")
    print("CIFAR-100 dataset loaded.")

    # Normalize the data
    train_data = train_data / 255.0
    test_data = test_data / 255.0

    # One-hot encode the labels
    train_labels_one_hot = np.zeros((train_labels.shape[0], 100))
    train_labels_one_hot[np.arange(train_labels.shape[0]), train_labels.astype(int)] = 1
    test_labels_one_hot = np.zeros((test_labels.shape[0], 100))
    test_labels_one_hot[np.arange(test_labels.shape[0]), test_labels.astype(int)] = 1

    # Mini-batch size
    batch_size = 600

    # Number of epochs
    num_epochs = 100

    # Learning rate
    learning_rate = 0.1

    # Initialize the modules
    # using the following network architecture: Linear(32 ∗ 32 ∗ 3, 100), SoftMax and the CrossEntropy criterion
    linear = Linear(32 * 32 * 3, 100)
    softmax = SoftMax()
    network = [linear, softmax]
    loss = CrossEntropy()

    train_and_test(
        network,
        loss,
        train_data,
        train_labels_one_hot,
        test_data,
        test_labels_one_hot,
        batch_size,
        num_epochs,
        learning_rate,
    )


if __name__ == "__main__":
    print("Test 2.d")
    print("========")
    #test_2_d()
    print("\n")
    print("Test 2.e.1")
    print("========")
    #test_2_e_1()
    print("\n")
    print("Test 2.e.2.coarse")
    print("========")
    test_2_e_2(use_coarse_labels=True)
    print("\n")
    print("Test 2.e.2.fine")
    print("========")
    test_2_e_2(use_coarse_labels=False)
