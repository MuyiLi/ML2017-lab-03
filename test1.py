from PIL import Image
import pylab
import numpy as np
import os
from feature import NPDFeature
import pickle

def save(model, filename):
    with open(filename, "wb") as f:
        pickle.dump(model, f)

pos_img_path = './datasets/original/nonface/'
#pos_img_path = '.\\datasets\\original\\nonface'
#pos_img_path = '/Users/limuyi/Downloads/study/大三上/机器学习/6班谭明奎全英班/机器学习实验/机器学习实验内容/3AdaBoost人脸分类/ML2017-lab-03/datasets/original/nonface'
features_face = []
pos_list_dir = os.listdir(pos_img_path)
count = 1
for filename in pos_list_dir:
	im = Image.open(pos_img_path+filename).convert('L').resize((24,24)) 
	im_array = np.array(im) 
	im_feature = NPDFeature(im_array).extract()
	print('pos:',count)
	count += 1
	features_face.append(im_feature)

save(features_face,'features_nonface')  
