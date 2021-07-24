from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import copy
import random
import time
import glob
from PIL import Image

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from transformer_model import VisionTransformer

dic = {
    '000': 0, '001': 1, '002': 2, '003': 3, '004': 4,
}
train_paths = glob.glob('../new_angle_data/train/*/*.npy')
test_paths = glob.glob('../new_angle_data/development/*/*.npy')
random.shuffle(train_paths)
max_acc = 0.0


def weight_init(m):
    if isinstance(m, nn.Linear):
        # 2种方式不一样
        # nn.init.xavier_normal_(m.weight)
        # nn.init.constant_(m.bias, 0)
        m.weight.data.normal_(0, 0.02)
        m.bias.data.zero_()
    # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv2d):
        # 3种方式不一样
        # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # nn.init.xavier_normal_(m.weight)
        m.weight.data.normal_(0, 0.02)
        m.bias.data.zero_()
    # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        # 2种方式不一样
        # nn.init.constant_(m.weight, 1)#全部初始化为1
        # nn.init.constant_(m.bias, 0)#全部初始化为0
        m.weight.data.normal_(1.0, 0.02)  # 全部初始化为均值为1，方差为0的正态分布
        m.bias.data.fill_(0)  # 全部初始化为0


def numpy2torch(data):
    x = data[:, :, :, :, 0]
    x = torch.from_numpy(x)
    x = x.to(torch.float32)
    x = x.permute(0, 2, 3, 1)
    x = x.reshape(data.shape[0], x.shape[1], -1)
    x.squeeze()
    return x


def tsting_and_save(net, net_path):
    correct = 0
    total = 0
    inputs = torch.zeros((len(test_paths), 128, 51))
    labels = torch.zeros((len(test_paths)))
    for index, path in enumerate(test_paths):
        path = path.replace('\\', '/')
        data = np.load(path)
        data = numpy2torch(data)
        inputs[index] = data
        labels[index] = dic[path.split('/')[-2]]
    with torch.no_grad():
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    global max_acc
    if max_acc < acc:
        max_acc = acc
        state = {'model': copy.deepcopy(net.state_dict()),
                 'optimizer': copy.deepcopy(optimizer.state_dict())}
        torch.save(state, net_path)

    print('Accuracy of the network on the %d development set: %f%%' % (total,
                                                                       100 * correct / total))


def training(net, criterion, optimizer, net_path, batch_size):
    inputs = torch.zeros((batch_size, 128, 51))
    labels = torch.zeros((batch_size))
    print('start training')
    s = time.time()
    for epoch in range(5):

        start = 0
        batches = 0
        for start in range(0, len(train_paths), batch_size):
            size = batch_size
            if len(train_paths) - batch_size * batches < batch_size:
                size = len(train_paths) - batch_size * batches
            for index in range(start, start + size):
                train_paths[index] = train_paths[index].replace('\\', '/')
                data = np.load(train_paths[index])
                data = numpy2torch(data)
                inputs[index - start] = data
                labels[index - start] = dic[train_paths[index].split('/')[-2]]

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            batches += 1

            print('epoch:%d batches:%d loss: %f' % (epoch + 1, batches, loss.item()))
        net.eval()
        tsting_and_save(net, net_path)

        e = time.time()
        print('time %0.2f min' % ((e - s) / 60))

    print('finish training')


if __name__ == '__main__':
    net_path = 'transformer_ResNet_11-15.pth'
    net = VisionTransformer(num_patches=128)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    net.apply(weight_init)

    # checkpoint = torch.load('transformer_ResNet_1-10.pth')
    # net.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])

    training(net, criterion, optimizer, net_path, batch_size=10)
