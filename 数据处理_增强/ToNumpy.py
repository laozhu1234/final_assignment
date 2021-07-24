# -*- coding: utf-8 -*-
'''
因为一些视频不是128帧的,通过RGB的reshape"到128帧"然后反解码(通过angle_data提供的坐标边界信息)回到五维度的npy
'''


def main():
    import numpy as np
    import glob
    import os
    import matplotlib.pyplot as plt
    import torchvision.transforms as transforms
    from PIL import Image
    
    
    def get_min_max(video_path):
        video = np.load(video_path)
        video[:, 1, :, :, :] = 0
        if np.sum(video[:, :, :, :, 1] != 0) == 0:
            video = video[:, :, :, :, 0]
        else:
            video = video.reshape((1, 3, video.shape[2], 34), order='F')
    
        video_min = video.min(axis=2).min(axis=2)   #帧数-关节点最小值,即最小的xyz坐标
        video_max = video.max(axis=2).max(axis=2)
        return video_min.squeeze(), video_max.squeeze()
    
    def transform_to_numpy(RGB_path):
        RGB_img = Image.open(RGB_path)
        RGB_img = RGB_img.convert("RGB")
        # RGB_img.show()
        resize = transforms.Resize([17, 128])
        RGB_img = resize(RGB_img)
        # RGB_img.show()
        video = np.array(RGB_img).astype(float)
        video_path = RGB_path.replace('\\', '/').replace('RGB', 'angle_data').replace('png', 'npy') #保证对应性
        video_min, video_max = get_min_max(video_path)
        for channel in range(video.shape[2]):
            if channel == 1:
                video[:, :, channel] = 0
            else:
                video[:, :, channel] = video[:, :, channel] * (video_max[channel] - video_min[channel]) / 255 + video_min[channel]
        video = np.transpose(video, (2, 1, 0))
        video = np.expand_dims(video, axis=0)
        video = np.expand_dims(video, axis=4)
        x = np.zeros(video.shape)
        video = np.concatenate((video, x), axis=4)
        video = video.astype(float)
        return video
    
    def sort_numpy():
        RGB_paths = glob.glob('../RGB/*/*/*.png')
        video_paths = glob.glob('../angle_data/*/*/*.npy')
    
        for RGB_path in RGB_paths:
            video = transform_to_numpy(RGB_path)
            
            video_path = RGB_path.replace('\\', '/').replace('RGB', 'new_angle_data').replace('png', 'npy')
            classes = video_path.split('/')[-2]
            label = video_path.split('/')[-3]
            file_path = '../new_angle_data/' + label + '/' + classes
            if not os.path.exists(file_path):
                os.makedirs(file_path)
                
            np.save(video_path, video)
    
    sort_numpy()
    
if __name__ == '__main__':
    main()
