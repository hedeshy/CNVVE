from torch import nn
from torchsummary import summary

import torch.nn.functional as F


class CNN2D(nn.Module):

    def __init__(self, n_output=6, stride=1, padding="same"):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d( in_channels=64, out_channels=128, kernel_size=3, stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d( in_channels=128, out_channels=128, kernel_size=3, stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d( in_channels=128, out_channels=128, kernel_size=3, stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()

        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(2304, n_output)
        self.softmax = nn.Softmax(dim=1)
    

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.dropout(x)
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions


class PiczakNet(nn.Module):
    def __init__(self, num_classes = 6):
        super(PiczakNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=80, kernel_size=(57, 6), stride=(1, 1), padding=(0, 3)),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 3), stride=(4, 3))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=80, out_channels=80, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        )
        
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(5600, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions



if __name__ == "__main__":

    model = CNN2D()
    summary(model.to('cuda'), (1, 96, 63))


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    n = count_parameters(model)
    print("Number of parameters: %s" % n)