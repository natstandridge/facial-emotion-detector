from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import datasets
import torchvision
from loader import FER2013

train = FER2013(train=True, transform=ToTensor())
dataset = DataLoader(train, batch_size=256, num_workers=16)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 42 * 42, 7) ## 7 is for the number of classes (emotions)
        )

    def forward(self, x):
        return(self.model(x))

def train(epochs=2):
    n = Net()

    ''' 
    Adam is a stochastic gardient descent optimizer that implements adaptive learning rates.
    The learning rate gets adjusted for each parameter based on its history of gradients.
        
    Adam takes the neural net parameters, and a learning rate.
    (higher learning rate causes parameters to be updated more aggressively)
    '''
    optimize = Adam(n.parameters(), lr=0.001)

    '''
    A loss function calculates how well the model is performing (predictions vs correct labels)
    The error is then used to update the model's parameters to make future predictions better
    '''
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print(f"Starting Epoch number {epoch}")
        for batch in dataset:
            print(f"Starting batch:\n {batch}")
            data, label = batch
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