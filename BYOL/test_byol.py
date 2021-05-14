import torch.nn as nn
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import torchvision
import copy

import argparse

import torch.optim as optim
from torchlars import LARS

from BYOL import BYOL

EPOCHS = 1000
image_size = 224
# image_size = 224
device = 'cuda'
BATCH = 1024
LR = 0.02 

parser = argparse.ArgumentParser()
parser.add_argument('--file', )
args = parser.parse_args()


class Eval(nn.Module):
    
    #def __init__(self, model): 
        # super(Eval, self).__init__()
        # self.model = model
        # self.model.fc = nn.Linear(2048, 10)
        # self.model.fc.requires_grad = True
    
    
    def __init__(self, model): 
        super(Eval, self).__init__()
        self.model = model
        self.fc = nn.Linear(512,10)
        self.fc.requires_grad = True
    
    def forward(self, x):
        out = self.model(x)
        out = out.reshape([-1,512])
        out = self.fc(out)

        return out

if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((224, 224)),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    # train_dir = '/home/imlab/tiny-imagenet-200/train'
    # trainset = torchvision.datasets.ImageFolder(train_dir, transform=transforms.ToTensor())
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH, shuffle=True)

    # val_dir = '/home/imlab/tiny-imagenet-200/val'
    # testset = torchvision.datasets.ImageFolder(val_dir, transform=transforms.ToTensor())
    # testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH, shuffle=False)


    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH,
                                              shuffle=False, num_workers=2)



    backbone = models.resnet18(pretrained=False)
    # backbone = models.resnet50(pretrained=False)
    # backbone.load_state_dict(torch.load("ebyol.ckpt"))
    #backbone.load_state_dict(torch.load("model-final.pth"))
    #backbone.load_state_dict(torch.load("model-20.pt"))

    byol = BYOL(net=backbone, projection_size=256, projection_hidden_size=4096, image_size=image_size).to(device)
    byol.load_state_dict(torch.load(args.file))

    model = copy.deepcopy(byol.net)
        
    for param in model.parameters():
        param.requires_grad = False

    model = Eval(model).to(device)
    model = nn.DataParallel(model)

    #optimizer = LARS(optim.SGD(model.parameters(), lr=LR, momentum=0.9))
    optimizer = optim.Adam(model.parameters(), lr=3e-3)
    criterion = nn.CrossEntropyLoss()

    for step in range(EPOCHS):
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            print('[%d, %5d] loss: %.3f' %
                  (step + 1, i + 1, loss))
            running_loss = 0.0

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = model(images.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum().item()

            print('Accuracy: %.3f %%' % (100 * correct / total))