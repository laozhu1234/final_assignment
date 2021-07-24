# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 09:36:44 2021

@author: Lenovo_ztk

训练集增强后命名为"原名_角度_是否翻转.npy",例xxx_67_1指旋转67°后水平镜像
"""
import numpy as np
import glob, shutil
import os
import math,random



def create_validation_set():
    #从首尾各取10%作验证集
    classes = ['000','001','002','003','004']
    for i in range(len(classes)):
        paths = glob.glob('../data/train/' + classes[i] +'/*.npy')
        
        vali_path = '../data/development/' + classes[i]
        if not os.path.exists(vali_path):
            os.makedirs(vali_path)
            
        for j in range(len(paths)//10):
            target = paths[j].replace('train','development')
            shutil.move(paths[j],target)
            target = paths[-(j+1)].replace('train','development')
            shutil.move(paths[-(j+1)],target)

        
if __name__ == '__main__':         
    create_validation_set() #将train中每一类分出20%到development中作验证集
    
    import random_rotateflip
    random_rotateflip.main()
    import ToRGB
    ToRGB.main()
    import ToNumpy
    ToNumpy.main()
    #随机旋转或镜像
    #转换成3*约128*17的RGB图像
    #利用RGB图像的resize将视频扩充到128帧,符合transformer的输入     

