import numpy as np

# 指定npy文件路径
file_path = "/home0/students/master/2022/wangzy/Pycharm-Remote(161)/WSSS_Model2/datasets/deepglobe/cls_labels_onehot_deepglobe.npy"

# 读取npy文件
data = np.load(file_path, allow_pickle=True)

print(data)
