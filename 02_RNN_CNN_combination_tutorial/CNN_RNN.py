# Reference : CNN + RNN - Concatenate time distributed CNN with LSTM (https://discuss.pytorch.org/t/solved-concatenate-time-distributed-cnn-with-lstm/15435/2)

import torch

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark=False

torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

class CNN_RNN(nn.Module):

    def __init__(self, device, rnn_type='lstm', bidirection='False', hidden_size=100, num_layers=2, learning_rate=0.0001):

        super().__init__()

        ### Deep Neural Network Setup ###
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.leakyrelu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.leakyrelu2 = nn.LeakyReLU(0.1)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.leakyrelu3 = nn.LeakyReLU(0.1)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm4 = nn.BatchNorm2d(256)
        self.leakyrelu4 = nn.LeakyReLU(0.1)
        
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.batchnorm5 = nn.BatchNorm2d(512)
        self.leakyrelu5 = nn.LeakyReLU(0.1)

        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm6 = nn.BatchNorm2d(512)
        self.leakyrelu6 = nn.LeakyReLU(0.1)

        self.conv7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.batchnorm7 = nn.BatchNorm2d(512)
        self.leakyrelu7 = nn.LeakyReLU(0.1)

        self.conv8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm8 = nn.BatchNorm2d(512)
        self.leakyrelu8 = nn.LeakyReLU(0.1)

        self.conv9 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.batchnorm9 = nn.BatchNorm2d(1024)
        self.leakyrelu9 = nn.LeakyReLU(0.1)

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn_type = rnn_type

        if rnn_type == 'rnn':
            self.RNN = nn.RNN(input_size=30720, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.5)

        elif rnn_type == 'lstm':
            self.RNN = nn.LSTM(input_size=30720, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.5)

        # self.linear = nn.Linear(in_features=hidden_size, out_features=6)
        self.linear = nn.Linear(in_features=hidden_size, out_features=3)

        ### Training Setup ###
        self.device = device
        self.to(self.device)
        
        self.optimizer = optim.RMSprop(self.parameters(), lr=learning_rate)

        self.translation_loss = nn.MSELoss()
        # self.rotation_loss = nn.MSELoss()

    def forward(self, x):

        batch_size, timestep, channel, height, width = x.size()

        # print('Original shape : {}'.format(x.shape))

        ### Tensor reshaping for CNN ###        
        x = x.view(batch_size * timestep, channel, height, width)       # Align time step into batch dimension in order to apply same CNN for every image
        # print('Modified shape for CNN: {}'.format(x.shape))

        ### Feature Extraction with CNN ###
        CNN_output = self.conv1(x)
        CNN_output = self.batchnorm1(CNN_output)
        CNN_output = self.leakyrelu1(CNN_output)

        CNN_output = self.conv2(CNN_output)
        CNN_output = self.batchnorm2(CNN_output)
        CNN_output = self.leakyrelu2(CNN_output)

        CNN_output = self.conv3(CNN_output)
        CNN_output = self.batchnorm3(CNN_output)
        CNN_output = self.leakyrelu3(CNN_output)

        CNN_output = self.conv4(CNN_output)
        CNN_output = self.batchnorm4(CNN_output)
        CNN_output = self.leakyrelu4(CNN_output)

        CNN_output = self.conv5(CNN_output)
        CNN_output = self.batchnorm5(CNN_output)
        CNN_output = self.leakyrelu5(CNN_output)

        CNN_output = self.conv6(CNN_output)
        CNN_output = self.batchnorm6(CNN_output)
        CNN_output = self.leakyrelu6(CNN_output)

        CNN_output = self.conv7(CNN_output)
        CNN_output = self.batchnorm7(CNN_output)
        CNN_output = self.leakyrelu7(CNN_output)

        CNN_output = self.conv8(CNN_output)
        CNN_output = self.batchnorm8(CNN_output)
        CNN_output = self.leakyrelu8(CNN_output)

        CNN_output = self.conv9(CNN_output)
        CNN_output = self.batchnorm9(CNN_output)
        CNN_output = self.leakyrelu9(CNN_output)
        # print('CNN output shape : {}'.format(CNN_output.shape))

        ### Tensor reshaping for RNN ###
        CNN_output = CNN_output.view(batch_size, timestep, -1)
        # print('Modified shape for RNN: {}'.format(CNN_output.shape))

        ### Training with RNN elements ###
        h0 = torch.zeros(self.num_layers, CNN_output.size(0), self.hidden_size).to(self.device)

        if self.rnn_type == 'rnn':            
            RNN_output, _ = self.RNN(CNN_output, h0)

        elif self.rnn_type == 'lstm':
            c0 = torch.zeros(self.num_layers, CNN_output.size(0), self.hidden_size).to(self.device)
            RNN_output, _ = self.RNN(CNN_output, (h0, c0))

        # print('RNN output shape : {}'.format(RNN_output.shape))

        ### Dense layer for 6 DOF estimation ###
        pose_est = self.linear(RNN_output[:, -1, :].view(batch_size, -1))
        # print('Final output shape : {}'.format(pose_est.shape))

        return pose_est
