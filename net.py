import torch
from torch import nn, save
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms
from tqdm import tqdm
import logging

from loader import FER2013

logging.basicConfig(filename="net.log", level=logging.INFO)

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomPerspective(0.25, 0.25),
    transforms.RandomErasing(0.25, (0.05, 0.25), (0.05, 0.25), 0, True)
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5]),
])

FER2013_train = FER2013(train = True, transform = train_transform)
FER2013_val = FER2013(train = False, transform = val_transform)

train_loader = DataLoader(FER2013_train, batch_size = 512, shuffle = True, pin_memory = True, num_workers = 8, multiprocessing_context = 'fork')
val_loader = DataLoader(FER2013_val, batch_size = 512, shuffle = False, pin_memory = True, num_workers = 8, multiprocessing_context = 'fork')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    '''
    Neural Net Structure:

        class Net(nn.Module):
            def __init__(self, num_channels, classes):
                super().__init__()
                self.conv1 = nn.Conv2d(num_channels, 32, (3, 3), padding = 1)
                self.bn1 = nn.BatchNorm2d(32)
                nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
                self.relu1 = nn.ReLU()
                self.conv2 = nn.Conv2d(32, 64, (3, 3), padding = 1)
                self.bn2 = nn.BatchNorm2d(64)
                nn.init.kaiming_uniform_(self.conv2.weight, mode = 'fan_in', nonlinearity='relu')
                self.relu2 = nn.ReLU()
                self.conv3 = nn.Conv2d(64, 128, (3, 3), padding = 1)
                self.bn3 = nn.BatchNorm2d(128)
                self.relu3 = nn.ReLU()
                self.conv4 = nn.Conv2d(128, 64, kernel_size = 1)
                self.bn4 = nn.BatchNorm2d(64)
                nn.init.kaiming_uniform_(self.conv1.weight, mode = 'fan_in', nonlinearity='relu')
                self.relu4 = nn.ReLU()
                self.conv5 = nn.Conv2d(64, classes, kernel_size = 1)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu1(x)
                x = self.conv2(x)
                x = self.bn2(x)
                x = self.relu2(x)
                x = self.conv3(x)
                x = self.bn3(x)
                x = self.relu3(x)
                x = self.conv4(x)
                x = self.bn4(x)
                x = self.relu4(x)
                x = self.conv5(x)

                return(x)

    Training Structure:

        def train(epochs = 1):
            n = Net(1, 7) ## one input channel (grayscale), 7 classes (emotions)
            n = n.to(device)
            train_loss = 0.0
            optimizer = Adam(n.parameters(), lr = 0.001)
            scheduler = StepLR(optimizer, step_size = 5, gamma = 0.1)
            loss_function = nn.CrossEntropyLoss()

            n.train()

            logging.info("Starting training...")
            
            for epoch in tqdm(range(epochs), desc = 'Epochs'):
                for data, label in tqdm(train_loader, desc = 'Training'):
                    data, label = data.to(device), label.to(device)
                    output = n(data).to(device)
                    loss = loss_function(output, label)

                    ## back propogation starts here
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step

                    train_loss += loss.item()

            ## start validation
            n.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for data, label in tqdm(val_loader, desc = 'Validation'):
                    data, label = data.to(device), label.to(device)
                    output = n(data).to(device)
                    loss = loss_function(output, label)
                    val_loss += loss.item()

                    total += label.size(0)     ## get number of samples in current batch and add to total
                    _, predicted = torch.max(output.data, 1)    ## ignore max value, get predicted label to compare to ground truth
                    correct += (predicted == label).sum().item()    ## compare predicted to ground truth, .sum() to count all True bools, .item() converts from single element tensor to a Python int/float 

            train_loss /= len(train_loader)         ## average training loss
            val_loss /= len(val_loader)             ## average validation loss
            val_accuracy = 100 * correct / total    ## percentage of correctly identified label instances

            print(f"Training and validation complete.\nEpochs completed: {epoch + 1} - Training loss: {train_loss:.5f} - Validation loss: {val_loss:.5f} - Validation accuracy: {val_accuracy:.5f}") ## add 1 to epoch when displaying to avoid counting from 0
            logging.info(f"Training and validation complete.\nEpochs completed: {epoch + 1} - Training loss: {train_loss:.5f} - Validation loss: {val_loss:.5f} - Validation accuracy: {val_accuracy:.5f}\nFootprint:\n{Net.__doc__}\n\n") ## storing structure in log to optimizer accuracy TODO: replace this with CSV storage once model is performing well

            with open('model.pt', 'wb') as f:
                save(n.state_dict(), f)

    '''
    def __init__(self, num_channels, classes):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, (3, 3), padding = 1)
        self.bn1 = nn.BatchNorm2d(32)
        nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, (3, 3), padding = 1)
        self.bn2 = nn.BatchNorm2d(64)
        nn.init.kaiming_uniform_(self.conv2.weight, mode = 'fan_in', nonlinearity='relu')
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 128, (3, 3), padding = 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 64, kernel_size = 1)
        self.bn4 = nn.BatchNorm2d(64)
        nn.init.kaiming_uniform_(self.conv1.weight, mode = 'fan_in', nonlinearity='relu')
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(64, classes, kernel_size = 1)
        self.avg_pool = nn.AvgPool2d(kernel_size = 48) ## reduce it down to 1D

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)

        return(x)

def train(epochs = 3):
    n = Net(1, 7) ## one input channel (grayscale), 7 classes (emotions)
    n = n.to(device)
    train_loss = 0.0
    optimizer = Adam(n.parameters(), lr = 0.001)
    scheduler = StepLR(optimizer, step_size = 5, gamma = 0.1)
    loss_function = nn.CrossEntropyLoss()

    n.train()

    logging.info("Starting training...")
    
    for epoch in tqdm(range(epochs), desc = 'Epochs'):
        for data, label in tqdm(train_loader, desc = 'Training'):
            data, label = data.to(device), label.to(device)
            output = n(data).to(device)
            loss = loss_function(output, label)

            ## back propogation starts here
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step

            train_loss += loss.item()

    ## start validation
    n.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, label in tqdm(val_loader, desc = 'Validation'):
            data, label = data.to(device), label.to(device)
            output = n(data).to(device)
            loss = loss_function(output, label)
            val_loss += loss.item()

            total += label.size(0)     ## get number of samples in current batch and add to total
            _, predicted = torch.max(output.data, 1)    ## ignore max value, get predicted label to compare to ground truth
            correct += (predicted == label).sum().item()    ## compare predicted to ground truth, .sum() to count all True bools, .item() converts from single element tensor to a Python int/float 

    train_loss /= len(train_loader)         ## average training loss
    val_loss /= len(val_loader)             ## average validation loss
    val_accuracy = 100 * correct / total    ## percentage of correctly identified label instances

    print(f"Training and validation complete.\nEpochs completed: {epoch + 1} - Training loss: {train_loss:.5f} - Validation loss: {val_loss:.5f} - Validation accuracy: {val_accuracy:.5f}") ## add 1 to epoch when displaying to avoid counting from 0
    logging.info(f"Training and validation complete.\nEpochs completed: {epoch + 1} - Training loss: {train_loss:.5f} - Validation loss: {val_loss:.5f} - Validation accuracy: {val_accuracy:.5f}\nFootprint:\n{Net.__doc__}\n\n") ## storing structure in log to optimizer accuracy TODO: replace this with CSV storage once model is performing well

    with open('model.pt', 'wb') as f:
        save(n.state_dict(), f)

if __name__ == '__main__':
    train()