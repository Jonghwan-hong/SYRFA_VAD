import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import random
from functools import partial

import numpy as np


class AdversarialLayer(torch.autograd.Function):
    iter_num = 0
    max_iter = 20000

    @staticmethod
    def forward(ctx, input):
        # self.iter_num += 1
        #         ctx.save_for_backward(iter_num, max_iter)
        AdversarialLayer.iter_num += 1
        return input * 1.0

    @staticmethod
    def backward(ctx, gradOutput):
        import pdb
        alpha = 10
        low = 0.0
        high = 1.0
        lamb = 2.0
        iter_num, max_iter = AdversarialLayer.iter_num, AdversarialLayer.max_iter
        # print('iter_num {}'.format(iter_num))
        coeff = np.float(lamb * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)
        # pdb.set_trace()
        return -coeff * gradOutput


class pretext_classifier(nn.Module):
    def __init__(self, ):
        super().__init__()

        # self.ad_layer1 = nn.Linear(feature_len, 1024)
        # self.ad_layer1.weight.data.normal_(0, 0.01)
        # self.ad_layer1.bias.data.fill_(0.0)
        # self.fc1 = nn.Sequential(self.ad_layer1, nn.ReLU(), nn.Dropout(0.5))

        self.ad_layer2 = nn.Linear(1024, 49)
        # self.ad_layer2.weight.data.normal_(0, 0.01)
        # self.ad_layer2.bias.data.fill_(0.0)
        # self.ad_layer3 = nn.Linear(512, 81)
        # self.ad_layer3.weight.data.normal_(0, 0.3)
        # self.ad_layer3.bias.data.fill_(0.0)
        # self.fc2_3 = nn.Sequential(self.ad_layer2, nn.ReLU(), self.ad_layer3)
        # self.adv = AdversarialLayer()

    def forward(self, x):
        # op_out = torch.bmm(y.unsqueeze(2), x.unsqueeze(1))
        # ad_in = op_out.view(-1,  y.size(1) * x.size(1))
        # f2 = self.fc1(x)
        # f = self.fc2_3(x)
        f = self.ad_layer2(x)

        return f


class discriminator(nn.Module):
    def __init__(self, ):
        super().__init__()

        # self.ad_layer1 = nn.Linear(feature_len, 1024)
        # self.ad_layer1.weight.data.normal_(0, 0.01)
        # self.ad_layer1.bias.data.fill_(0.0)
        # self.fc1 = nn.Sequential(self.ad_layer1, nn.ReLU(), nn.Dropout(0.5))

        self.ad_layer2 = nn.Linear(1024, 2)
        # self.ad_layer2.weight.data.normal_(0, 0.01)
        # self.ad_layer2.bias.data.fill_(0.0)
        # self.ad_layer3 = nn.Linear(512, 2)
        # self.ad_layer3.weight.data.normal_(0, 0.3)
        # self.ad_layer3.bias.data.fill_(0.0)
        # self.fc2_3 = nn.Sequential(self.ad_layer2, nn.ReLU(), self.ad_layer3)

    def forward(self, x):
        # op_out = torch.bmm(y.unsqueeze(2), x.unsqueeze(1))
        # ad_in = op_out.view(-1,  y.size(1) * x.size(1))
        # f2 = self.fc1(x)
        # f = self.fc2_3(x)
        f = self.ad_layer2(x)
        return f


class MappingNetwork(torch.nn.Module):
    def __init__(self, depth=5):
        super().__init__()
        self.depth = depth
        self.weight1 = nn.ParameterList()
        self.bias1 = nn.ParameterList()
        self.weight2 = nn.ParameterList()
        self.bias2 = nn.ParameterList()
        self.weight3 = nn.ParameterList()
        self.bias3 = nn.ParameterList()
        self.weight4 = nn.ParameterList()
        self.bias4 = nn.ParameterList()

        for i in range(depth):
            self.weight1.append(nn.Parameter(torch.ones((32, 7, 32, 32))))
            self.bias1.append(nn.Parameter(torch.zeros((32, 7, 32, 32))))

            self.weight2.append(nn.Parameter(torch.ones((64, 7, 16, 16))))
            self.bias2.append(nn.Parameter(torch.zeros((64, 7, 16, 16))))

            self.weight3.append(nn.Parameter(torch.ones((64, 8, 8))))
            self.bias3.append(nn.Parameter(torch.zeros((64, 8, 8))))

            # self.weight4.append(nn.Parameter(torch.ones((512,1,7,7))))
            # self.bias4.append(nn.Parameter(torch.zeros((512,1,7,7))))

        self.relu = nn.ReLU(inplace=True)

    def fea1(self, x):
        input = x
        for i in range(self.depth - 1):
            x = self.relu(self.weight1[i] * x + self.bias1[i])
        x = self.weight1[i + 1] * x + self.bias1[i + 1]
        return x

    def fea2(self, x):
        input = x
        for i in range(self.depth - 1):
            x = self.relu(self.weight2[i] * x + self.bias2[i])
        x = self.weight2[i + 1] * x + self.bias2[i + 1]
        return x

    def fea3(self, x):
        input = x
        for i in range(self.depth - 1):
            try:
                x = self.relu(self.weight3[i] * x + self.bias3[i])
            except Exception:
                x = self.relu(self.weight3[i] * x + self.bias3[i])
        x = self.weight3[i + 1] * x + self.bias3[i + 1]
        return x

    # def fea4(self, x):
    #     for i in range(self.depth - 1):
    #         x = self.relu(self.weight4[i] * x + self.bias4[i])
    #     x = self.weight4[i + 1] * x + self.bias4[i + 1]
    #     return x


class Adaparams(nn.Module):
    def __init__(self, depth=10):
        super(Adaparams, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.depth = depth
        self.weight = nn.ParameterList()
        self.bias = nn.ParameterList()
        for i in range(depth):
            self.weight.append(nn.Parameter(torch.ones(1024)))
            self.bias.append(nn.Parameter(torch.zeros(1024)))

    def forward(self, x):
        for i in range(self.depth - 1):
            x = self.relu(self.weight[i] * x + self.bias[i])
        x = self.weight[i + 1] * x + self.bias[i + 1]
        return x


class WideBranchNet(nn.Module):

    def __init__(self, time_length=7, num_classes=[81, 81]):
        super(WideBranchNet, self).__init__()

        self.time_length = time_length
        self.num_classes = num_classes
        self.isaug = True
        self.layer1 = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)))
        self.layer2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)))
        self.layer3 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(self.time_length, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)))

        self.conv2d = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Dropout2d(p=0.3)
        )

        self.max2d = nn.MaxPool2d(2, 2)

    def mixstyle(self, x):
        alpha = 0.1
        beta = torch.distributions.Beta(alpha, alpha)
        B = x.size(0)
        mu = x.mean(dim=[3, 4], keepdim=True)
        var = x.var(dim=[3, 4], keepdim=True)
        sig = (var + 1e-6).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig
        lmda = beta.sample((B, 1, 1, 1, 1))
        lmda = lmda.to(x.device)
        perm = torch.randperm(B)
        mu2, var2 = mu[perm], var[perm]
        # mu2 = style.mean(dim=[3,4], keepdim=True)
        # var2 = style.var(dim=[3,4], keepdim=True)
        sig2 = (var2 + 1e-6).sqrt()
        try:
            mu_mix = mu * lmda + mu2 * (1 - lmda)
        except Exception:
            mu_mix = mu * lmda + mu2 * (1 - lmda)
        sig_mix = sig * lmda + sig2 * (1 - lmda)
        return x_normed * sig_mix + mu_mix

    def fea1(self, x):
        x = self.layer1(x)

        N, C, T, H, W = x.size()
        if random.random() > 0.5:
            self.isaug = True
            aug_x = self.mixstyle(x)
            # aug_x = self.mixstyle(x.permute(0,2,1,3,4).reshape(-1,C,H,W))
            # aug_x = aug_x.reshape(N,T,C,H,W).permute(0,2,1,3,4)

        else:
            self.isaug = False
            aug_x = x
        return x, aug_x

    def fea2(self, x, aug_x):
        x = self.layer2(x)
        N, C, T, H, W = x.size()
        aug_x = self.layer2(aug_x)

        if not self.isaug:
            aug_x = self.mixstyle(aug_x)
            # aug_x = self.mixstyle(aug_x.permute(0,2,1,3,4).reshape(-1,C,H,W))
            # aug_x = aug_x.reshape(N,T,C,H,W).permute(0,2,1,3,4)
        return x, aug_x

    def fea3(self, x):
        x = self.layer3(x)
        x = x.squeeze(2)
        return x

    def flat(self, x):
        x = self.max2d(self.conv2d(x))
        x = x.view((x.size(0), -1))
        return x

    def fea_forward(self, x):
        x = self.fea3(x)
        x = self.flat(x)
        return x

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out3 = out3.squeeze(2)

        return out3


if __name__ == '__main__':
    net = WideBranchNet(time_length=9, num_classes=[49, 81])
    x = torch.rand(2, 3, 7, 64, 64)
    out = net(x)
