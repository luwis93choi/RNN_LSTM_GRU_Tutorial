# Reference : PyTorch Tutorial - RNN & LSTM & GRU - Recurrent Neural Nets (https://www.youtube.com/watch?v=0_PgWWmauHk)

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Input Shape for RNN : N x 28 x 28 (Batch x Sequence Length x Data shape)
# => Each batch has 28 sequences of 28 feature vector
#    N batch of 28 sequence of 28 feature vector

input_size = 28
sequence_length = 28

RNN_layer_num = 2
RNN_hidden_size = 256

num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)  # RNN works with any size of sequence length
        # Batch First : Batch Size x Time Sequence x Data shape

        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)
        # hidden_size * sequence_length = Concat the entire length of data into RNN ---> Use the info from every hidden state of all the sequence

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Hidden state shape : RNN Layer Num x Batch Size x Hidden Size

        # Foward propagation
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)     # Flatten RNN output (Batch size x Flattened Data shape)
                                                # Use all the hidden state output from all the sequence
        out = self.fc(out)

        return out

class RNN_Last_Hidden_to_FC(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_Last_Hidden_to_FC, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)  # RNN works with any size of sequence length
        # Batch First : Batch Size x Time Sequence x Data shape

        self.fc = nn.Linear(hidden_size, num_classes)
        # Use only the last hidden state output for Fully Connected Layer Regression

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Hidden state shape : RNN Layer Num x Batch Size x Hidden Size

        # Foward propagation
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])      # Use only the last hidden state output from all the sequence

        return out

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRU, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)  # GRU works with any size of sequence length
        # Batch First : Batch Size x Time Sequence x Data shape

        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)
        # hidden_size * sequence_length = Concat the entire length of data into GRU ---> Use the info from every hidden state of all the sequence

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Hidden state shape : GRU Layer Num x Batch Size x Hidden Size

        # Foward propagation
        out, _ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)     # Flatten GRU output (Batch size x Flattened Data shape)
                                                # Use all the hidden state output from all the sequence
        out = self.fc(out)

        return out

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  # LSTM works with any size of sequence length
        # Batch First : Batch Size x Time Sequence x Data shape

        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)
        # hidden_size * sequence_length = Concat the entire length of data into LSTM ---> Use the info from every hidden state of all the sequence

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Hidden state shape : LSTM Layer Num x Batch Size x Hidden Size

        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Cell state shape : LSTM Layer Num x Batch Size x Hidden Size

        # Foward propagation
        out, _ = self.lstm(x, (h0, c0))
        out = out.reshape(out.shape[0], -1)     # Flatten LSTM output (Batch size x Flattened Data shape)
                                                # Use all the hidden state output from all the sequence
        out = self.fc(out)

        return out

class LSTM_Last_Hidden_to_FC(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM_Last_Hidden_to_FC, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  # LSTM works with any size of sequence length
        # Batch First : Batch Size x Time Sequence x Data shape

        self.fc = nn.Linear(hidden_size, num_classes)
        # Use only the last hidden state output for Fully Connected Layer Regression

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Hidden state shape : LSTM Layer Num x Batch Size x Hidden Size

        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Cell state shape : LSTM Layer Num x Batch Size x Hidden Size

        # Foward propagation
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])      # Use only the last hidden state output from all the sequence

        return out

train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model_selection = 4

if model_selection == 0:
    RNN_model = RNN(input_size, RNN_hidden_size, RNN_layer_num, num_classes).to(device)

elif model_selection == 1:
    RNN_model = RNN_Last_Hidden_to_FC(input_size, RNN_hidden_size, RNN_layer_num, num_classes).to(device)

elif model_selection == 2:
    RNN_model = GRU(input_size, RNN_hidden_size, RNN_layer_num, num_classes).to(device)

elif model_selection == 3:
    RNN_model = LSTM(input_size, RNN_hidden_size, RNN_layer_num, num_classes).to(device)

elif model_selection == 4:
    RNN_model = LSTM_Last_Hidden_to_FC(input_size, RNN_hidden_size, RNN_layer_num, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(RNN_model.parameters(), lr=learning_rate)

for epcoh in range(num_epochs):

    for batch_idx, (data, targets) in enumerate(train_loader):
        
        # Pytorch Dataloader data output shape : Batch x 1 x Data shape (MNIST : Batch Size x 1 x 28 x 28)

        # Required data shape for RNN : Batch x Sequnece Length x Data shape (MNIST : Batch Size x 28 x 28)

        # Since each 28 x 28 MNIST image will be broken down along the row, each image can be turned into 28 sequence of 28 data vector.
        # Width 28 x Height 28 MNIST image will become Sequence 28 x Feature 28

        # DataLoader output's 2nd dimension needs to be removed.

        # For general usage, dataloader for RNN-based model needs to stack the data for the length of sequence before inserting into the model

        print('Dataloader Original Output Shape : {}'.format(data.size()))
        data = data.to(device=device).squeeze(1)    # Remove dimension 1
        print('Dataloader Rectified Output Shape : {}'.format(data.size()))
        targets = targets.to(device=device)
        print('--------------------------')

        scores = RNN_model(data)
        loss = criterion(scores, targets)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0

    # Set model to eval
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with \
              accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )
    # Set model back to train
    model.train()


check_accuracy(train_loader, RNN_model)
check_accuracy(test_loader, RNN_model)