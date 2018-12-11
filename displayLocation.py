import cv2 as cv
from matplotlib import pyplot as plt
import os



def displayLocation(locationIndex):
	img = cv.imread('youAreHere/'+str(locationIndex)+'.jpg',1)
	plt.imshow(img),plt.show()
def displayLocation(locationIndex,inputLocation):
	img = cv.imread('youAreHere/'+str(locationIndex)+'.jpg',1)
	img2 = cv.imread('youAreHere/'+str(inputLocation)+'.jpg',1)
	plt.subplot(121),plt.imshow(img),plt.title("predicted location")
	plt.subplot(122),plt.imshow(img2),plt.title("input location")
	plt.show()



leniancy_dict = {1:[2,3],2:[1,3,4],3:[1,2,4],4:[2,3,5],5:[4,6],6:[5,7],7:[6,8,9],8:[7],9:[7,10],10:[9,11],11:[10],12:[13],13:[12,14],14:[13,15],15:[14,16,17],16:[15],17:[15,18],18:[17,19],19:[18,20],20:[19,21],21:[20,22],22:[21]}

