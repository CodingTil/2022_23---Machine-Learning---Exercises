import torch.nn as nn
import torch.nn.functional as F
import functools
import operator


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #####Insert your code here for subtask 1d#####
        # Define building blocks of CNNs: convolution and pooling layers
        # CNN with 3 layers (one layer: convolution (5x5) + ReLU + pooling (3x3 block, stide size 2)), and 24, 32, 50 filters, stride size 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=self.conv1.out_channels,
            out_channels=32,
            kernel_size=5,
            stride=1,
        )
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(
            in_channels=self.conv2.out_channels,
            out_channels=50,
            kernel_size=5,
            stride=1,
        )
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        #####Insert your code here for subtask 1e#####
        # Define fully connected layers
        # 3 fully connected layers with 100, 50, 3
        self.fc1 = nn.Linear(
            in_features=450,
            out_features=100,
        )
        self.fc2 = nn.Linear(in_features=self.fc1.out_features, out_features=50)
        self.fc3 = nn.Linear(in_features=self.fc2.out_features, out_features=3)

    def forward(self, x, label):
        """Run forward pass for the network
        :param x: a batch of input images -> Tensor
        :param label: a batch of GT labels -> Tensor
        :return: loss: total loss for the given batch, logits: predicted logits for the given batch
        """

        #####Insert your code here for subtask 1f#####
        # Feed a batch of input image x to the main building blocks of CNNs
        # Do not forget to implement ReLU activation layers here
        y = x
        y = self.conv1(y)
        y = F.relu(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = F.relu(y)
        y = self.pool2(y)
        y = self.conv3(y)
        y = F.relu(y)
        y = self.pool3(y)

        #####Insert your code here for subtask 1g#####
        # Feed the output of the building blocks of CNNs to the fully connected layers
        y = y.view(y.size(0), -1)
        y = self.fc1(y)
        y = self.fc2(y)
        y = self.fc3(y)

        #####Insert your code here for subtask 1h#####
        # Implement cross entropy loss on the top of the output of softmax
        logits = F.softmax(y, dim=1)
        loss = F.cross_entropy(logits, label)

        return loss, logits
