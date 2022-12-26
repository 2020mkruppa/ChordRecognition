from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Lambda
from torch.optim.lr_scheduler import MultiStepLR
import pickle
import numpy as np
import FeatureExtractor as FE


conv_layer_half_width = 4

class AudioDataset(Dataset):
    def __init__(self, source):
        self.transform = ToTensor()
        self.target_transform = Lambda(lambda y: torch.zeros(len(FE.chordDict), dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
        with open(source, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        # 4 channels
        # For each channel, 10x12 "image" of frame, like chromagram, but with cycles distinct
        inputNP = np.reshape(np.array(entry[0:4:2], dtype=np.single), (2, 10, 12))
        return torch.from_numpy(inputNP), entry[4]
    
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)
        self.dropout4 = nn.Dropout(0.5)
        
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.batchnorm4 = nn.BatchNorm1d(32)
        
        self.fc1 = nn.Linear(10*12*2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, len(FE.chordDict))

    def forward(self, x): #Input is (batchNum, 2, 10, 12)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.batchnorm1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.batchnorm2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = F.relu(x)
        x = self.batchnorm3(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        x = F.relu(x)
        x = self.dropout4(x)
        
        x = self.fc5(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_correct = 0
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        output = model(data)
        loss = F.nll_loss(output, target, reduction='sum')
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        train_correct += pred.eq(target.view_as(pred)).sum().item()
    
    train_loss /= len(train_loader.dataset)
    print('Train Epoch: {} \tLoss: {:.6f}, Accuracy: {}/{} ({:.0f}%)'.format(epoch, train_loss,
                                           train_correct, len(train_loader.dataset), 100. * train_correct / len(train_loader.dataset)))


def test(model, device, loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(loader.dataset), 100. * correct / len(loader.dataset)))

#From https://github.com/pytorch/examples/blob/main/mnist/main.py
def main():
    train_batch_size = 128
    epochs = 30
    learningRate = 0.0003
    gamma = 0.3
              
    torch.manual_seed(1)
    device = torch.device("cpu")
   
    train_loader = torch.utils.data.DataLoader(AudioDataset('trainData.pickle'), batch_size=train_batch_size)
    validation_loader = torch.utils.data.DataLoader(AudioDataset('validationData.pickle'))
    #test_loader = torch.utils.data.DataLoader(AudioDataset('testData.pickle'))

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learningRate, weight_decay=0.001) #0.001
    
    print(str(sum(p.numel() for p in model.parameters() if p.requires_grad)) + " learnable parameters")
    
    scheduler = MultiStepLR(optimizer, milestones=[8, 16], gamma=gamma)
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, validation_loader)
        #test(model, device, test_loader)
        scheduler.step()
        
        #if epoch == 9:
        #    torch.save(model.state_dict(), "model9.pt")


if __name__ == '__main__':
    main()
    
    
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=(3, (2*conv_layer_half_width) + 1), stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, (2*conv_layer_half_width) + 1), stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, (2*conv_layer_half_width) + 1), stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, (2*conv_layer_half_width) + 1), stride=1, padding=0)
        
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.batchnorm4 = nn.BatchNorm2d(32)
        
        self.fc1 = nn.Linear(32*12*2, 64)
        self.fc2 = nn.Linear(64, len(FE.chordDict))

    def forward(self, x): #Input is (batchNum, 4, 10, 12)
        x = F.pad(x, (conv_layer_half_width, conv_layer_half_width, 0, 0), mode='circular') #Want side padding for wrapping notes, but not top-bottom wrap
        #Pads only on the last axis so we preserve having length 12 and can do circular convolution, need extra 0s for pytorch bug
        x = self.conv1(x)
        x = F.relu(x)
        x = self.batchnorm1(x)
        x = F.pad(x, (conv_layer_half_width, conv_layer_half_width, 0, 0), mode='circular') 
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchnorm2(x)
        x = F.pad(x, (conv_layer_half_width, conv_layer_half_width, 0, 0), mode='circular')
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchnorm3(x)
        x = F.pad(x, (conv_layer_half_width, conv_layer_half_width, 0, 0), mode='circular')
        
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchnorm4(x)
        
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
'''