
import torch.nn as nn
import torch

class CustomBis(nn.Module):

    def __init__(self, im_size=(3, 56, 56)):
        super(CustomBis, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=(2, 2)) 
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(2, 2)) 
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3)) 
        self.conv5 = nn.Conv2d(64, 64, kernel_size=(3, 3)) 
        self.conv6 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=(3, 3))
        self.conv8 = nn.Conv2d(128, 256, kernel_size=(3, 3))
        self.conv9 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=2) 
        self.conv10 = nn.Conv2d(512, 512, kernel_size=(3, 3))
        self.conv11 = nn.Conv2d(512, 1024, kernel_size=(4, 4)) 

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

        self.conv_layer = nn.Sequential(
            self.conv1, self.relu,
            self.conv2, self.relu,
            # self.conv3, self.relu,
            self.conv4, self.relu,
            self.conv5, self.relu,
            self.conv6, self.relu,
            self.conv7, self.relu,
            self.conv8, self.relu,
            self.conv9, self.relu,
            self.conv10, self.relu,
            self.conv11, self.relu
        )

        out_shape = self.get_flattened_output_shape(im_size)

        self.fc1 = nn.Linear(out_shape, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, 1)

        self.fc_layer = nn.Sequential(
            self.fc1,
            self.fc2, 
            self.fc3, 
            self.fc4
        )

    def get_flattened_output_shape(self, im_size):
        rd_im = torch.randn((1,) + im_size)
        output = self.conv_layer(rd_im)
        output = output.reshape(output.shape[0], -1)
        return output.shape[1]

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc_layer(x)
        return self.sigmoid(x)

     