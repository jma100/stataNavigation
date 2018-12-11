import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import time
import os
import logging
from displayLocation import displayLocation, leniancy_dict
logging.basicConfig(format='%(asctime)s %(message)s')

orb = cv.ORB_create()

# compute and store features
root_dir = "C:/Users/Mairead/Desktop/6.819/stataNavigation/data/train/"

postfix = ['_L_W.JPG', '_L_E.JPG', '_L_S.JPG', '_P_N.jpg', '_P_S.jpg', '_P_NW.jpg', '_P_NE.jpg', '_P_W.jpg', '_P_E.jpg', '_P_SW.jpg', '_P_SE.jpg', '_L_N.JPG']

def index2location(idx):
    return int(idx/12) + 1

def index2file(idx):
    return str(int(idx / 12) + 1) + postfix[idx % 12]

start = time.time()

input_location = input("pick a number 1-22:\n>")
folder_input_location = input_location
if int(folder_input_location) < 10:
    folder_input_location = "0"+input_location

test_img = cv.imread('data/val/'+folder_input_location+'/02.jpg',0)
test_img = cv.resize(test_img, (0,0), fx=0.2, fy=0.2)
test_kp, test_des = orb.detectAndCompute(test_img,None)
features = np.load("featuresOrb.npy",encoding="latin1")
sum_list = []
#22 locations, 12 images per location
assert len(features)==22*12

distance = 45
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

for idx, des in enumerate(features):
    matches = bf.match(test_des,des)
    good = list(filter(lambda x:x.distance < distance, matches))
    sum_list.append(len(good))

best_idx = sum_list.index(max(sum_list))
fn = index2file(best_idx)
location_idx = index2location(best_idx)

print("time taken:",time.time()-start)
displayLocation(location_idx,input_location)
