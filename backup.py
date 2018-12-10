import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import time
import os
import logging
logging.basicConfig(format='%(asctime)s %(message)s')

sift = cv.xfeatures2d.SIFT_create()

# # compute and store features
# root_dir = "/home/jingweim/jingweim/819/temp/stataNavigation/data/train/"
# features = []

# # np.save("features.npy", features)

# for i in range(1,23):
# 	folder_name = "%02d" % i + "/"
# 	images = os.listdir(root_dir+folder_name)
# 	for name in images:
# 		img = cv.imread(root_dir+folder_name+name,0)
# 		kp, des = sift.detectAndCompute(img, None)
# 		features.append(des)
# 	print("folder " + folder_name + " done!")
# 		# import pdb; pdb.set_trace()

# np.save("features.npy", features)
# import pdb; pdb.set_trace()

postfix = ['_L_W.JPG', '_L_E.JPG', '_L_S.JPG', '_P_N.jpg', '_P_S.jpg', '_P_NW.jpg', '_P_NE.jpg', '_P_W.jpg', '_P_E.jpg', '_P_SW.jpg', '_P_SE.jpg', '_L_N.JPG']

def index2file(idx):
	return str(idx / 12 + 1) + postfix[idx % 12]

start = time.time()

# load features:
test_img = cv.imread('data/val/13/01.jpg',0)
test_kp, test_des = sift.detectAndCompute(test_img,None)
features = np.load("features.npy")
sum_list = []
assert len(features)==22*12

for idx, des in enumerate(features):
	bf = cv.BFMatcher()
	matches = bf.knnMatch(test_des,des, k=2)
	good = []
	for m,n in matches:
	    if m.distance < 0.75*n.distance:
	        good.append([m])
	sum_list.append(len(good))
print(time.time()-start)

best_idx = sum_list.index(max(sum_list))
print(index2file(best_idx))