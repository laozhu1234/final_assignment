import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import confusion_matrix

torch.manual_seed(0)
net_path = 'ResNet_model.pth'
BATCH_SIZE = 50  # 批处理尺寸(batch_size)


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=5):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 3, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 4, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 6, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18():
    return ResNet(ResidualBlock)


def tst(net, testloader, save_name, var1_path):
    with torch.no_grad():
        correct = 0.0
        total = 0.0
        T_labels = 0.0
        T_predicted = 0.0
        for i, data in enumerate(testloader, 0):
            net.eval()
            images, labels = data
            images, labels = images.to(), labels.to()
            outputs = net(images)
            # 取得分最高的那个类 (outputs.data的索引号)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            if i == 0:
                T_labels = labels
                T_predicted = predicted
            else:
                T_labels = torch.cat((T_labels, labels))
                T_predicted = torch.cat((T_predicted, predicted))
            # print("predicted =", predicted)

            correct += (predicted == labels).sum()
        cm = confusion_matrix(T_labels, T_predicted)
        # print("T_label =\n", T_labels)
        # print("T_predicted =\n", T_predicted)
        # print(outputs)
        print(cm)
        print('test acc：%.3f%%' % (100 * correct / total))


def main():
    # 准备数据集并预处理
    transform = transforms.Compose([
        transforms.Resize((17, 128)),
        # transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dvl_dataset = datasets.ImageFolder(root='../RGB/development', transform=transform)
    dvl_loader = DataLoader(dvl_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_dataset = datasets.ImageFolder(root='../RGB/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 模型定义-ResNet
    net = ResNet18().to()

    checkpoint = torch.load(net_path)
    net.load_state_dict(checkpoint['model'])
    print("————————————————————-————————————DEVELOPMENT SET———————————————————————-—————————")
    tst(net, dvl_loader, "R_dvl_pred.npy", "R_dvl_label.npy")
    print("————————————————————-————————————TEST SET———————————————————————-—————————")
    tst(net, test_loader, "R_test_pred.npy", "R_test_label.npy")
    print("————————————————————————————————————FINISH————————————————————————————————————")


if __name__ == '__main__':
    main()
