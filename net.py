import torch
from torch import nn, save, flatten
from torch.nn import LogSoftmax
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms
from tqdm import tqdm

from loader import FER2013

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

FER2013_data = FER2013(train = True, transform = transform) ## turn to tensor and normalize pixel values
train_ratio = 0.8

num_training_samples = int(len(FER2013_data) * train_ratio)
num_val_samples = len(FER2013_data) - num_training_samples
train_dataset, val_dataset = random_split(FER2013_data, [num_training_samples, num_val_samples])

train_loader = DataLoader(train_dataset, batch_size = 256, shuffle = True, pin_memory = True, num_workers = 0)  ## have to set num_workers 0 now for some reason
val_loader = DataLoader(val_dataset, batch_size = 256, shuffle = False, pin_memory = True, num_workers = 0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self, num_channels, classes):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, (5, 5), padding = 2)
        nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))
        self.conv2 = nn.Conv2d(32, 128, (5, 5), padding = 2)
        nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))
        self.conv3 = nn.Conv2d(128, 128, (3, 3), padding = 1)
        self.relu3 = nn.ReLU()
        ##self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(18432, 64)
        nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(64, classes)
        self.logSM = LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)

        return(self.logSM(x))

def train(epochs = 3):
    n = Net(1, 7) ## one input channel (grayscale), 7 classes (emotions)
    n = n.to(device)
    training_loss = 0.0

    ''' 
    Adam is a stochastic gradient descent optimizer that implements adaptive learning rates.
    The learning rate gets adjusted for each paramters based on its history of gradients.
        
    Adam takes the neural net parameters, and a learning rate.
    (higher learning rate causes parameters to be updated more aggressively)
    '''
    optimize = Adam(n.parameters(), lr = 0.0001)

    '''
    A loss function calculates how well the model is performing (predictions vs correct labels)
    The error is then used to update the model's parameters to make future predictions better
    '''
    loss_function = nn.CrossEntropyLoss()

    for epoch in tqdm(range(epochs)):
        for data, label in tqdm(train_loader):
            data, label = data.to(device), label
            output = n(data).to(device)
            loss = loss_function(output, label)

            ## back propogation starts here
            optimize.zero_grad()
            loss.backward()
            optimize.step()

            training_loss += loss.item()

        ## start validation
        n.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, label in tqdm(val_loader):
                data, label = data.to(device), label
                output = n(data).to(device)
                loss = loss_function(output, label)
                val_loss += loss.item()

                _, predicted = torch.max(output.data, 1)        ## ignore max value, get predicted label to compare to ground truth
                total += label.size(0)                          ## get number of samples in current batch and add to total

                correct += (predicted == label).sum().item()    ## compare predicted to ground truth, .sum() to count all True bools, .item() converts from single element tensor to a Python int/float 

        train_loss /= len(train_loader)         ## average training loss
        val_loss /= len(val_loader)             ## average validation loss
        val_accuracy = 100 * correct / total    ## percentage of correctly identified label instances

        print(f"Epoch: {epoch + 1} - Training loss: {training_loss:.5f} - Validation loss: {val_loss:.5f} - Validation accuracy: {val_accuracy:.5f}") ## add 1 to epoch when displaying to avoid counting from 0

    with open('model.pt', 'wb') as f:
        save(n.state_dict(), f)

if __name__ == '__main__':
    train()