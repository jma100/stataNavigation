# PlacesCNN for scene classification
#
# by Bolei Zhou
# last modified by Bolei Zhou, Dec.27, 2017 with latest pytorch and torchvision (upgrade your torchvision please if there is trn.Resize error)

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
import torch.nn as nn
import time
import numpy as np
from collections import Counter 

start = time.time()
# th architecture to use
# arch = 'resnet18' #0.54
arch = 'alexnet' #0.13
# arch = 'resnet50' #0.41
# arch = 'densenet161' #0.54

# load the pre-trained weights
model_file = '%s_places365.pth.tar' % arch
if not os.access(model_file, os.W_OK):
    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
    os.system('wget ' + weight_url)

model = models.__dict__[arch](num_classes=365)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model = nn.Sequential(*list(model.children())[:-1])
model.eval()


# load the image transformer
centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# root_dir = "/home/jingweim/jingweim/819/temp/stataNavigation/data/train/"
# features = []
# for i in range(1,23):
#     folder_name = "%02d" % i + "/"
#     images = os.listdir(root_dir+folder_name)
#     for name in images:
#         # img = Image.open(root_dir+folder_name+name).convert('L')
#         img = Image.open(root_dir+folder_name+name)

#         # import pdb; pdb.set_trace()
#         input_img = V(centre_crop(img).unsqueeze(0))
#         # input_img = torch.cat((input_img, input_img, input_img), 1)
#         logit = model.forward(input_img)
#         new = logit.view(-1)
#         features.append(new.detach().numpy())
#         # import pdb; pdb.set_trace()
#     print("folder " + folder_name + " done!")
#       # import pdb; pdb.set_trace()

# np.save("features_pretrained_grey.npy", np.array(features))
# print(time.time()-start)
# import pdb; pdb.set_trace()
postfix = ['_L_W.JPG', '_L_E.JPG', '_L_S.JPG', '_P_N.jpg', '_P_S.jpg', '_P_NW.jpg', '_P_NE.jpg', '_P_W.jpg', '_P_E.jpg', '_P_SW.jpg', '_P_SE.jpg', '_L_N.JPG']


def index2file(idxs):
    # import pdb; pdb.set_trace()
    result = ""
    for idx in idxs:
        result += str(idx.item() / 12 + 1) + postfix[idx % 12] + "\n"
    return result

def getMode(idxs):
    result = []
    for idx in idxs:
        result.append(idx.item() / 12 + 1)
    return Counter(result).most_common(1)[0][0]

# load the test image
features = np.load("features_pretrained_alexnet.npy")
val_dir = "/home/jingweim/jingweim/819/temp/stataNavigation/data/val/"

# val_image = val_dir+"03/02.jpg"
# img = Image.open(val_image)
# input_img = V(centre_crop(img).unsqueeze(0))
# logit = model.forward(input_img)
# new = logit.view(-1).detach().numpy()

# dist = []
# for i in features:
#     dist.append(np.linalg.norm(i-new))
# import pdb; pdb.set_trace()
correct = 0.0
for i in range(1,23):
    start_time = time.time()
    folder_name = "%02d" % i + "/"
    # images = os.listdir(test_dir+folder_name)
    images = ["02.jpg"]
    for name in images:
        # img_name = 'data/val/13/01.jpg'
        # img = Image.open(val_dir+folder_name+name).convert('L')
        img = Image.open(val_dir+folder_name+name)

        input_img = V(centre_crop(img).unsqueeze(0))
        # input_img = torch.cat((input_img, input_img, input_img), 1)
        # forward pass
        logit = model.forward(input_img)
        new = logit.view(-1).detach().numpy()
        test = torch.from_numpy(new).expand((264,9216))
        train = torch.from_numpy(features)
        l2 = torch.sqrt(torch.sum(torch.mul(train - test, train - test), dim=1))
        copy = l2
        probs, idx = l2.sort(0, False)


        # import pdb; pdb.set_trace()
        print("####################################")
        # print(probs)
        print(name)
        file = index2file(idx[:5])
        print(file)
        mode = getMode(idx[:5])
        if i == mode or i == mode-1 or i == mode+1:
        # if i == idx[0]/12+1:
            correct+=1
            print("yes")

        orientation = copy[(mode-1)*12: mode *12]
        probs1, idx1 = orientation.sort(0, False)
        prediction = postfix[idx1[0]]
        prediction = prediction[prediction.index("_")+1:prediction.index(".")]
        prediction = prediction[prediction.index("_")+1:]
        print("location: "+ str(mode))
        print("orientation: "+ prediction)
    print(time.time()-start)
print(correct/22)
