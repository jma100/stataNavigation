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
features = []

# # # # np.save("features.npy", features)

# for i in range(1,23):
#   folder_name = "%02d" % i + "/"
#   images = os.listdir(root_dir+folder_name)
#   for name in images:
#       img = cv.imread(root_dir+folder_name+name,0)
#       img = cv.resize(img, (0,0), fx=0.2, fy=0.2)
#       kp, des = orb.detectAndCompute(img, None)
#       features.append(des)
#   print("folder " + folder_name + " done!")
#       # import pdb; pdb.set_trace()

# np.save("featuresOrb.npy", features)
# # import pdb; pdb.set_trace()

postfix = ['_L_W.JPG', '_L_E.JPG', '_L_S.JPG', '_P_N.jpg', '_P_S.jpg', '_P_NW.jpg', '_P_NE.jpg', '_P_W.jpg', '_P_E.jpg', '_P_SW.jpg', '_P_SE.jpg', '_L_N.JPG']

def index2location(idx):
    return int(idx/12) + 1

def index2file(idx):
    return str(int(idx / 12) + 1) + postfix[idx % 12]

start = time.time()



for numToMatch in range(1,13):
    correct_img = 0
    correct_location = 0
    for i in range(1,23):

        # load features:
        if i < 10:
            test_img = cv.imread('data/val/0'+str(i)+'/02.jpg',0)
        else: 
            test_img = cv.imread('data/val/'+str(i)+'/02.jpg',0)
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

        ##################################
        #Trying to match more than one image at the location
        location_totals = []
        k = 1
        location_sums = []
        for img_sum in sum_list:
            location_sums.append(img_sum)
            if k%12 == 0:
                #Find three best matches per location
                location_sum = 0
                for j in range(numToMatch):
                    location_sum += max(location_sums)
                    location_sums.remove(max(location_sums))
                location_sums = []
                location_totals.append(location_sum)
            k+=1
        best_loaction = location_totals.index(max(location_totals))+1
        # print(best_loaction)

        best_idx = sum_list.index(max(sum_list))
        fn = index2file(best_idx)
        location_idx = index2location(best_idx)
        # displayLocation(location_idx)

        if fn[1] == "_":
            new_fn = '0'+fn
        # print(fn)

        # matches = sorted(matches, key=lambda x:x.distance)
        # best_img = cv.imread('data/train/'+new_fn[0:2]+'/'+fn,0)
        # best_img = cv.resize(best_img, (0,0), fx=0.2, fy=0.2)
        # best_kp, best_des = orb.detectAndCompute(best_img,None)
        # matchImg = cv.drawMatches(test_img, test_kp, best_img, best_kp,matches[:10], None,flags=2)
        # plt.imshow(matchImg),plt.show()

        # if((i < 10 and int(fn[0])==i) or (i >= 10 and fn[1] != "_" and int(fn[0:2])==i)):
        #     correct_img += 1
        # if(best_loaction == i  or best_loaction in leniancy_dict[i]):
        #     correct_location += 1
        if(best_loaction == i):
            correct_location += 1
        # if (i == location_idx or location_idx in leniancy_dict[i]):
        #     correct_img += 1


    # print("correct img: ",correct_img/22.)
    print("correct location: ",correct_location/22.)

print(time.time()-start)

# img2 = cv.drawKeypoints(test_img, test_kp, None, color=(0,255,0), flags=0)
# plt.imshow(img2), plt.show()
