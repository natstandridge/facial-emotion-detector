import torch
from torch import nn, save
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from loader import FER2013

train = FER2013(train = True, transform = ToTensor())
dataset = DataLoader(train, batch_size = 32, shuffle = True, pin_memory = True, num_workers = 16)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (5, 5), padding = 1), ## one input channel (because its grayscale)
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, (5, 5), padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, (5, 5), padding = 1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(4096, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )

    def forward(self, x):
        return(self.model(x))

def train(epochs = 50):
    n = Net()
    n = n.to(device)

    ''' 
    Adam is a stochastic gradient descent optimizer that implements adaptive learning rates.
    The learning rate gets adjusted for each paramters based on its history of gradients.
        
    Adam takes the neural net parameters, and a learning rate.
    (higher learning rate causes parameters to be updated more aggressively)
    '''
    optimize = Adam(n.parameters(), lr = 0.01)

    '''
    A loss function calculates how well the model is performing (predictions vs correct labels)
    The error is then used to update the model's parameters to make future predictions better
    '''
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print(f"Starting epoch {epoch}")
        for index, batch in enumerate(dataset):
            print(f"Starting batch {index}")
            data, label = batch
            data, label = data.to(device), label.to(device)
            yhat = n(data)
            loss = loss_function(yhat, label)

            ## back propogation begins
            optimize.zero_grad()
            loss.backward()
            optimize.step()

        print(f"Epoch: {epoch}, loss: {loss.item()}")

    with open('model.pt', 'wb') as f:
        save(n.state_dict(), f)

if __name__ == '__main__':
    train()