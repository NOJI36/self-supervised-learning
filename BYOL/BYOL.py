import torch.nn as nn
import math
import torch
from torchvision import transforms as T
import torch.nn.functional as F
import copy

import random

device = 'cuda'

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)


class Online(nn.Module):
    def __init__(self, net, projection_size=256, projection_hidden_size=4096):
        super(Online, self).__init__()
        self.net = net
        #self.projection = MLP(2048, projection_size, projection_hidden_size) #224
        self.projection = MLP(512, projection_size, projection_hidden_size) #64
        self.prediction = MLP(projection_size, projection_size, projection_hidden_size)

    def forward(self, x):
        representation = self.net(x)
        representation = representation.reshape(representation.shape[0], -1)
        projection = self.projection(representation)
        prediction = self.prediction(projection)

        return prediction


class Target(nn.Module):
    def __init__(self, net, projection_size=256, projection_hidden_size=4096):
        super(Target, self).__init__()
        self.net = net
        #self.projection = MLP(2048, projection_size, projection_hidden_size)
        self.projection = MLP(512, projection_size, projection_hidden_size)


    def forward(self, x):
        representation = self.net(x)
        representation = representation.reshape(representation.shape[0], -1)
        projection = self.projection(representation)

        return projection


class BYOL(nn.Module):
    def __init__(self, net, projection_size=256, projection_hidden_size=4096, moving_average_decay=0.99, image_size=224):
        super(BYOL, self).__init__()
        self.tau_base = moving_average_decay


        # self.DEFAULT_AUG = TransformsSimCLR(size=224)

        DEFAULT_AUG = torch.nn.Sequential(
            RandomApply(
                T.ColorJitter(0.8, 0.8, 0.8, 0.2),
                p=0.8
            ),
            T.RandomGrayscale(p=0.2),
            T.RandomHorizontalFlip(),
            # RandomApply(
            #     T.GaussianBlur((3, 3), (1.0, 2.0)),
            #     p=0.2
            # ),
            T.RandomResizedCrop((image_size, image_size)),
            # T.Normalize(
            #     mean=torch.tensor([0.485, 0.456, 0.406]),
            #     std=torch.tensor([0.229, 0.224, 0.225])),
        )

        self.augment1 = DEFAULT_AUG
        self.augment2 = DEFAULT_AUG

        self.net = nn.Sequential(*list(net.children())[:-1])

        self.online = Online(self.net, projection_size, projection_hidden_size).to(device)

        self.target = Target(copy.deepcopy(self.net), projection_size, projection_hidden_size).to(device)
        for param in self.target.parameters():
            param.requires_grad = False

    def EMA(self, step, EPOCHS):
        tau = 1 - (1 - self.tau_base) * (math.cos(math.pi * step / EPOCHS) + 1) / 2
        for param_target, param_online in zip(self.target.parameters(), self.online.parameters()):
            param_target.data = tau * param_target.data + (1 - tau) * param_online.data
            

    def loss_fn(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)

    def forward(self, x):
        image_one = self.augment1(x)
        image_two = self.augment2(x)

        online_pred_one = self.online(image_one)
        online_pred_two = self.online(image_two)

        with torch.no_grad():
            target_proj_one = self.target(image_one)
            target_proj_two = self.target(image_two)

        loss_one = self.loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = self.loss_fn(online_pred_two, target_proj_one.detach())

        loss = loss_one + loss_two

        return loss.mean()
