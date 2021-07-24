# -*- coding: utf-8 -*-
'''
足够精简的代码,xz坐标旋转数据增强
'''

'''
每个数据保留源数据,并旋转num_rotate次,据此得到的(1+num_rotate)份数据每份还可能以pr_flip的概率水平镜像
所以总共得到(1+num_rotate)*(1+pr_flip),例如train-000类最终期望数数据量为100*(1+2)*(1+0.5)=450

使用相似性修改,修改部分紧贴注释后的朱栩颉源码,以供对照
'''
def main():
    import numpy as np
    import glob, shutil
    import os
    import math,random
    
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    
    
    
    
    
    # angles_train={
    #     '000':[2 * math.pi * i / 5 for i in range(5)],
    #     '001':[2 * math.pi * i / 5 for i in range(5)],
    #     '002':[2 * math.pi * i / 5 for i in range(5)],
    #     '003':[2 * math.pi * i / 5 for i in range(5)],
    #     '004':[2 * math.pi * i / 50 for i in range(50)]
    # }
    
    # angles_test={
    #     '000':[2 * math.pi * i / 5 for i in range(5)],
    #     '001':[2 * math.pi * i / 5 for i in range(5)],
    #     '002':[2 * math.pi * i / 5 for i in range(5)],
    #     '003':[2 * math.pi * i / 5 for i in range(5)],
    #     '004':[2 * math.pi * i / 14 for i in range(14)]
    # }
    num_rotate={'000':2,'001':2,'002':2,'003':2,'004':15}
    pr_flip={'000':0.5,'001':0.5,'002':0.5,'003':0.5,'004':0.8}
    
    def rotate(video_path, angle):
        # 输入路径和旋转角度,返回旋转后的np
        video = np.load(video_path)
    
        angle_video = np.array(video)
        for index in range(video.shape[3]):     #如果有两个人,一并转了
            angle_video[:, 0, :, index] = video[:, 0, :, index] * math.cos(angle) + video[:, 2, :, index] * math.sin(
                angle)
            angle_video[:, 2, :, index] = -video[:, 0, :, index] * math.sin(angle) + video[:, 2, :, index
                                                                                     ] * math.cos(angle)
        return angle_video
    
    def sample(possibility):
        #根据输入的概率返回bool
        return random.random()<possibility
    
    def save_angle_numpy():
        sample_paths = glob.glob('../data/train/*/*.npy')
        for video_path in sample_paths:
            
            video_path = video_path.replace('\\', '/')
            name = video_path.split('/')[-1].split('.npy')[0]
            classes = video_path.split('/')[-2]
            label = video_path.split('/')[-3]
            file_path = video_path.split('/' + name)[0].replace('data', 'angle_data')
            if not os.path.exists(file_path):
                os.makedirs(file_path)
    
              # if label == 'train':
              #    for index, angle in enumerate(angles_train[classes]):
              #        x = rotate(video_path, angle)
              #        np.save(file_path + '/' + name + '_' + str(index) + '.npy', x)
              # else:
              #       for index, angle in enumerate(angles_test[classes]):
              #           x = rotate(video_path, angle)
              #           np.save(file_path + '/' + name + '_' + str(index) + '.npy', x)
            x = np.load(video_path)
            np.save(file_path+ '/'+ name+ '_0_0.npy', x)
            if(sample(pr_flip[classes])):
                x[:,0,:,:,:]=-x[:,0,:,:,:] #水平镜像
                np.save(file_path+ '/'+ name+ '_0_1.npy', x)
                
            for index in range(num_rotate[classes]):
                angle = int((random.random()-0.5)*180) #-90~90°随机抽样
                x = rotate(video_path,angle/180*math.pi)
                np.save(file_path+ '/'+ name+ '_'+str(angle)+'_0.npy', x)
                if(sample(pr_flip[classes])):
                    x[:,0,:,:,:]=-x[:,0,:,:,:] #水平镜像
                    np.save(file_path+ '/'+ name+ '_'+str(angle)+'_1.npy', x)
          
        #将验证集集复制一份到angle_data中,方便ToRGB调用
        development_path = '../data/development'
        new_path = '../angle_data/development'
        if os.path.exists(new_path):
            shutil.rmtree(new_path) #如果angle_data/development已存在则删除
        shutil.copytree(development_path,new_path)
        
        #将测试集复制一份到angle_data中,方便ToRGB调用
        test_path = '../data/test'
        new_path = '../angle_data/test'
        if os.path.exists(new_path):
            shutil.rmtree(new_path) #如果angle_data/test已存在则删除
        shutil.copytree(test_path,new_path)
            
    
    save_angle_numpy()

if __name__ == '__main__':
    main()