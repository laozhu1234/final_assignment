import glob
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

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
dvl_paths = glob.glob('../new_angle_data/development/*/*.npy')
test_paths = glob.glob('../new_angle_data/test/*/*.npy')
max_acc = 0.0


def numpy2torch(data):
    x = data[:, :, :, :, 0]
    x = torch.from_numpy(x)
    x = x.to(torch.float32)
    x = x.permute(0, 2, 3, 1)
    x = x.reshape(data.shape[0], x.shape[1], -1)
    x.squeeze()
    return x


def tst(net, testpath):
    correct = 0
    total = 0
    inputs = torch.zeros((len(testpath), 128, 51))
    T_labels = torch.zeros((len(testpath)))
    for index, path in enumerate(testpath):
        path = path.replace('\\', '/')
        data = np.load(path)
        data = numpy2torch(data)
        inputs[index] = data
        T_labels[index] = dic[path.split('/')[-2]]
    with torch.no_grad():
        outputs = net(inputs)
        _, T_predicted = torch.max(outputs.data, 1)
        total += T_labels.size(0)
        correct += (T_predicted == T_labels).sum().item()
        cm = confusion_matrix(T_labels, T_predicted)
        print(cm)
        print('test acc：%.3f%%' % (100 * correct / total))


if __name__ == '__main__':
    net_path = 'transformer_model.pth'
    net = VisionTransformer(num_patches=128)
    criterion = nn.CrossEntropyLoss()

    checkpoint = torch.load(net_path)
    net.load_state_dict(checkpoint['model'])
    print("————————————————————-————————————DEVELOPMENT SET———————————————————————-—————————")
    tst(net, dvl_paths)
    print("————————————————————-————————————TEST SET———————————————————————-—————————")
    tst(net, test_paths)
    print("————————————————————————————————————FINISH————————————————————————————————————")
