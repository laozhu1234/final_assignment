# -*- coding: utf-8 -*-
'''
序列2RGB,没有丢失信息
17*128的图像每个像素存放了xyz的相对位置(以框住整个人的正长方体框为基准)
只需知道每个文件的xyz的最大最小坐标就可以完全无损恢复
'''



def main():
    import numpy as np
    import glob
    import os
    import matplotlib.pyplot as plt
    import torchvision.transforms as transforms
    from PIL import Image

    def numpy2RGB(angle_path):
        data = np.load(angle_path)
        data[:, 1, :, :, :] = 0
        data = data[0, :, :, :, 0] #第一个人的,维度*帧数*关节点
        
        #得到所有帧所有关节最高和最低的xz点
        data_min = data.min(axis=1).min(axis=1)
        data_max = data.max(axis=1).max(axis=1)
    
        for channel in range(data.shape[0]):
            if channel == 1:    continue #指跳过第1维:y
            data[channel] = 255 * (data[channel] - data_min[channel]) / (data_max[channel] - data_min[channel])
    
        data = data.transpose(2, 1, 0)
        return np.floor(data)
    
    def save_RGB():
        angle_paths = sample_paths = glob.glob('../angle_data/*/*/*.npy')
        
        for path in angle_paths:
            path = path.replace('\\', '/')
            RGB_data = numpy2RGB(path)
            label = path.split('/')[-3]
            classes = path.split('/')[-2]
            file_path = '../RGB/' + label + '/' + classes
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            plt.imsave(path.replace('angle_data', 'RGB').replace('npy', 'png'), np.uint8(RGB_data))
    
    save_RGB()
    
if __name__ == '__main__':
    main()