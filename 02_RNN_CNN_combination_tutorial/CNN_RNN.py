# Reference : CNN + RNN - Concatenate time distributed CNN with LSTM (https://discuss.pytorch.org/t/solved-concatenate-time-distributed-cnn-with-lstm/15435/2)

import torch

# torch.manual_seed(42)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark=False

# torch.cuda.manual_seed(42)
# torch.cuda.manual_seed_all(42)

import os
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.models as models

import numpy as np

class CNN_RNN(nn.Module):

    def __init__(self, device, cnn_type='new', cnn_freeze=False, rnn_type='lstm', bidirection='False', regression='last_only', input_sequence_length=2, hidden_size=100, num_layers=2, learning_rate=0.0001):

        super().__init__()

        ### Deep Neural Network Setup ###
        self.cnn_type = cnn_type

        if cnn_type == 'new':
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

                nn.MaxPool2d(kernel_size=3, stride=1),

                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1),

                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1),

                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1),

                # nn.AvgPool2d(kernel_size=3, stride=1),

                nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1),

                nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1),

                nn.AvgPool2d(kernel_size=3, stride=1),

                nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(2048),
                nn.LeakyReLU(0.1),
            )

            CNN_flat_output_size = 16384
            self.CNN.to(device)

        elif cnn_type == 'mobilenetV3_large':

            self.CNN = models.mobilenet_v3_large(pretrained=True)

            CNN_flat_output_size = 1000

            if cnn_freeze == True:
                print('Freeze ' + cnn_type)
                for name, module in self.CNN.named_children():
                    for layer in module.children():
                        for param in layer.parameters():
                            param.requires_grad = False
                            # print(param)

        elif cnn_type == 'mobilenetV3_small':

            self.CNN = models.mobilenet_v3_small(pretrained=True)

            CNN_flat_output_size = 1000

            if cnn_freeze == True:
                print('Freeze ' + cnn_type)
                for name, module in self.CNN.named_children():
                    for layer in module.children():
                        for param in layer.parameters():
                            param.requires_grad = False
                            # print(param)

        elif cnn_type == 'vgg16':

            self.CNN = models.vgg16(pretrained=True)

            CNN_flat_output_size = 1000

            if cnn_freeze == True:
                print('Freeze ' + cnn_type)
                for name, module in self.CNN.named_children():
                    for layer in module.children():
                        for param in layer.parameters():
                            param.requires_grad = False
                            # print(param)

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn_type = rnn_type
        self.regression = regression

        if rnn_type == 'rnn':

            if num_layers > 1:
                self.RNN = nn.RNN(input_size=CNN_flat_output_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.5)

            else:
                self.RNN = nn.RNN(input_size=CNN_flat_output_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        elif rnn_type == 'lstm':

            if num_layers > 1:
                self.RNN = nn.LSTM(input_size=CNN_flat_output_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.5)

            else:
                self.RNN = nn.LSTM(input_size=CNN_flat_output_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)


        # self.linear = nn.Linear(in_features=hidden_size, out_features=6)
        
        if regression == 'last_only':
            self.linear = nn.Linear(in_features=hidden_size, out_features=3)    # For last sequence only case

        elif regression == 'full_sequence':
            in_features_num = input_sequence_length * hidden_size
            out_features_num = (num_layers * hidden_size)//2
            self.linear1 = nn.Linear(in_features=in_features_num, out_features=out_features_num)    # For full sequence
            self.batchnorm_linear1 = nn.BatchNorm1d(out_features_num)
            self.leakyrelu_linear1 = nn.LeakyReLU(0.1)
            self.dropout_linear1 = nn.Dropout(p=0.5)

            in_features_num = out_features_num
            out_features_num = out_features_num//2
            self.linear2 = nn.Linear(in_features=in_features_num, out_features=out_features_num)    # For full sequence
            self.batchnorm_linear2 = nn.BatchNorm1d(out_features_num)
            self.leakyrelu_linear2 = nn.LeakyReLU(0.1)
            self.dropout_linear2 = nn.Dropout(p=0.5)

            in_features_num = out_features_num
            out_features_num = 3
            self.linear3 = nn.Linear(in_features=in_features_num, out_features=out_features_num)    # For full sequence

        ### Training Setup ###
        self.device = device
        self.to(self.device)
        
        # self.optimizer = optim.RMSprop(self.parameters(), lr=learning_rate)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.translation_loss = nn.MSELoss()
        # self.rotation_loss = nn.MSELoss()

    def forward(self, x):

        batch_size, timestep, channel, height, width = x.size()

        # print('Original shape : {}'.format(x.shape))

        ### Tensor reshaping for CNN ###        
        x = x.view(batch_size * timestep, channel, height, width)       # Align time step into batch dimension in order to apply same CNN for every image
        # print('Modified shape for CNN: {}'.format(x.shape))

        ### Feature Extraction with CNN ###
        CNN_output = self.CNN(x)
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
        if self.regression == 'last_only':
            pose_est = self.linear(RNN_output[:, -1, :].view(batch_size, -1))     # Use the hidden state from the last LSTM sequence
        
        elif self.regression == 'full_sequence':
            full_RNN_hidden = torch.reshape(RNN_output, (batch_size, -1))
            # print('RNN output shape : {}'.format(full_RNN_hidden.shape))

            pose_est = self.linear1(full_RNN_hidden)     # Use the all hidden state from entire sequence of LSTM
            pose_est = self.batchnorm_linear1(pose_est)
            pose_est = self.leakyrelu_linear1(pose_est)
            pose_est = self.dropout_linear1(pose_est)

            pose_est = self.linear2(pose_est)
            pose_est = self.batchnorm_linear2(pose_est)
            pose_est = self.leakyrelu_linear2(pose_est)
            pose_est = self.dropout_linear2(pose_est)

            pose_est = self.linear3(pose_est)

        # print('Final output shape : {}'.format(pose_est.shape))

        return pose_est
