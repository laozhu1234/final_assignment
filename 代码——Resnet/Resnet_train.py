import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from Nadam import Nadam

# —————————————————————自主设置区—————————————————————
torch.manual_seed(0)
# 超参数设置
EPOCH = 10  # 遍历数据集次数
BATCH_SIZE = 50  # 批处理尺寸(batch_size)
LR = 0.0001  # 学习率
net_path = 'ResNet.pth'
ReTrain = False
Namda_flag = True


# —————————————————————自主设置区—————————————————————

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


def train(net, trainloader, optimizer, epoch, criterion):
    print('\nEpoch: %d' % (epoch + 1))
    net.train()
    sum_loss = 0.0
    correct = 0.0
    total = 0.0
    length = len(trainloader)
    for i, data in enumerate(trainloader, 0):
        # 准备数据
        inputs, labels = data
        inputs, labels = inputs.to(), labels.to()
        optimizer.zero_grad()

        # forward + backward
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 每训练1个batch打印一次loss和准确率
        sum_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()
        print('[epoch:%d|iter:%d|rate：%.2f%%] Loss: %.03f | Acc: %.2f%% '
              % (epoch + 1, (i + 1), 100. * ((i + 1) / length), sum_loss / (i + 1), 100. * correct / total))


def tst(net, testloader, optimizer, net_path, best_acc, best_epoch, epoch):
    print("Waiting Test!")
    with torch.no_grad():
        correct = 0
        total = 0
        for data in testloader:
            net.eval()
            images, labels = data
            images, labels = images.to(), labels.to()
            outputs = net(images)
            # 取得分最高的那个类 (outputs.data的索引号)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('test acc：%.3f%%' % (100 * correct / total))
        acc = 100. * correct / total
        if acc > best_acc:
            state = {'model': copy.deepcopy(net.state_dict()),
                     'optimizer': copy.deepcopy(optimizer.state_dict())}
            torch.save(state, net_path)
            best_acc = acc
            best_epoch = epoch

    return best_acc, best_epoch


def main():
    # 准备数据集并预处理
    transform = transforms.Compose([
        transforms.Resize((17, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = datasets.ImageFolder(root='../RGB/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_dataset = datasets.ImageFolder(root='../RGB/development', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # 模型定义-ResNet
    net = ResNet18().to()

    # 定义损失函数和优化方式
    criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
    # optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9,
    #                       weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    if Namda_flag:
        optimizer = optim.Adam(net.parameters(), lr=LR)
    else:
        optimizer = Nadam(net.parameters(), lr=LR)

    best_acc = 0  # 初始化best test accuracy
    best_epoch = 0  # 初始化best test accuracy的epoch
    if ReTrain:
        checkpoint = torch.load(net_path)
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    print("Start Training, Resnet-18!")
    T_s = time.time()
    best_acc, best_epoch = tst(net, test_loader, optimizer, net_path, best_acc, -1, best_epoch-1)
    for epoch in range(EPOCH):
        s = time.time()
        print("————————————————————————————————TRAIN START————————————————————————————————")
        train(net, train_loader, optimizer, epoch, criterion)
        print("————————————————————————————————TRAIN FINISH————————————————————————————————")
        print("")
        print("————————————————————-————————————TEST START———————————————————————-—————————")
        best_acc, best_epoch = tst(net, test_loader, optimizer, net_path, best_acc, best_epoch, epoch)
        e = time.time()
        print('[epoch:%d|进度:%.2f%%|用时:%.2fmin]'
              % (epoch + 1, (epoch + 1) / EPOCH, (e - s) / 60))
        print('[best_acc:%.2f%%|best_epoch:%d]'
              % (best_acc, (best_epoch + 1)))
        print("————————————————————————————————TEST FINISH————————————————————————————————")
    T_e = time.time()
    print("总耗时：", (T_e - T_s) / 60, "min")


if __name__ == '__main__':
    main()
