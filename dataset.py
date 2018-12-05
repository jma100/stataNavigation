import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt


import matplotlib as mpl
mpl.rcParams['axes.grid'] = False
mpl.rcParams['image.interpolation'] = 'nearest'
mpl.rcParams['figure.figsize'] = 15, 25


data_root = './data/'
train_root = data_root + 'train'
val_root = data_root + 'val'
test_root = data_root + 'test'

base_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.RandomCrop(128),
    transforms.ToTensor()
    # transforms.Normalize([0.5]*3, [0.5]*3)
    ])

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def show_dataset(dataset, n=6):
  img = np.vstack((np.hstack((np.asarray(dataset[i][0].permute(2,1,0)) for _ in range(n)))
                   for i in range(3)))
  plt.imsave("demo.png",img)
  plt.axis('off')

train_dataset = datasets.ImageFolder(root=train_root, transform=base_transform)
# train_dataset = ImageFolderWithPaths(root=train_root, transform=base_transform)

# import pdb; pdb.set_trace()
# show_dataset(train_dataset)

# import pdb; pdb.set_trace()
# val_dataset = datasets.ImageFolder(root=val_root, transform=base_transform)
# test_dataset = datasets.ImageFolder(root=test_root, transform=base_transform)

def get_data_loaders(batch_size):
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return (train_loader, val_loader)

def get_val_test_loaders(batch_size):
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return (val_loader, test_loader)

def get_train_loaders(batch_size):
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return train_loader


