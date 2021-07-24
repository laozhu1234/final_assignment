首先运行'./数据处理_增强/main.py' 
作验证集划分和训练集增强,此时'./'目录下会多出'./angle_data','./RGB','./new_angle_data'三个目录
其中'./angle_data'可以删除
'./RGB'将被Resnet调用
'./new_angle_data'将被transformer调用


然后分别运行'./代码——Resnet/Resnet_test.py'和'./代码——Transfer/transformer_test'即可查看本组训练所得最优模型的测试分类准确率(Resnet:74.359%,transformer:82.051%)