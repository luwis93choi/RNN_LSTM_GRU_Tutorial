# Reference : CNN + RNN architecture for video recognition (https://stackoverflow.com/questions/63567352/cnn-rnn-architecture-for-video-recognition)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

class CNN_RNN(nn.Module):

    def __init__(self, device, hidden_size=100, num_layers=2, learning_rate=0.0001):

        super(CNN_RNN, self).__init__()

        ### Deep Neural Network Setup ###
        self.CNN = nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
        )

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.RNN = nn.RNN(input_size=30720, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.linear = nn.Linear(in_features=hidden_size, out_features=6)

        ### Training Setup ###
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.translation_loss = nn.MSELoss()
        self.rotation_loss = nn.MSELoss()

        self.device = device
        self.to(self.device)

    def forward(self, x):

        batch_size, timestep, channel, height, width = x.size()

        # print('Original shape : {}'.format(x.shape))

        ### Tensor reshaping for CNN ###        
        x = x.view(batch_size * timestep, channel, height, width)
        # print('Modified shape for CNN: {}'.format(x.shape))

        ### Feature Extraction with CNN ###
        CNN_output = self.CNN(x)
        # print('CNN output shape : {}'.format(CNN_output.shape))

        ### Tensor reshaping for RNN ###
        CNN_output = CNN_output.view(batch_size, timestep, -1)
        # print('Modified shape for RNN: {}'.format(CNN_output.shape))

        ### Training with RNN elements ###
        h0 = torch.zeros(self.num_layers, CNN_output.size(0), self.hidden_size).to(self.device)

        RNN_output, _ = self.RNN(CNN_output, h0)
        # print('RNN output shape : {}'.format(RNN_output.shape))

        ### Dense layer for 6 DOF estimation ###
        pose_est = self.linear(RNN_output)
        # print('Final output shape : {}'.format(pose_est.shape))

        return pose_est
