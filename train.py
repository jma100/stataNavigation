import gc
import sys
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
torch.backends.cudnn.benchmark=True

import dataset
from models.AlexNet import *
from models.ResNet import *


def run():
    # Parameters
    num_epochs = 3
    output_period = 4
    batch_size = 4

    # setup the device for running
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet_18()
    model = model.to(device)

    train_loader= dataset.get_train_loaders(batch_size)
    num_train_batches = len(train_loader)

    criterion = nn.CrossEntropyLoss().to(device)
    # TODO: optimizer is currently unoptimized
    # there's a lot of room for improvement/different optimizers
    # optimizer = optim.SGD(model.parameters(), lr=1e-3) # original optimizer
    # optimizer = optim.SGD(model.parameters(), lr = 0.01) # bigger lr
    # optimizer = optim.SGD(model.parameters(), lr = 0.1) # even bigger lr

    optimizer = optim.SGD(model.parameters(), lr = 0.1, momentum=0.9, weight_decay=1e-4) # bigger lr with momentum and weight decay
    epoch = 1
    while epoch <= num_epochs:
        running_loss = 0.0
        for param_group in optimizer.param_groups:
            print('Current learning rate: ' + str(param_group['lr']))
        model.train()

        for batch_num, (inputs, labels) in enumerate(train_loader, 1):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

            if batch_num % output_period == 0:
                print('[%d:%.2f] loss: %.3f' % (
                    epoch, batch_num*1.0/num_train_batches,
                    running_loss/output_period
                    ))
                running_loss = 0.0
                gc.collect()

        gc.collect()
        # save after every epoch
        torch.save(model.state_dict(), "models/model.%d" % epoch)



        # model.eval()
        # # transformer = construct_transformer()
        # classification_count = 0
        # top5_count = 0
        # for batch_num, (inputs, labels) in enumerate(val_loader, 1):
        #     inputs = inputs.to(device)
        #     prediction = model(inputs)
        #     prediction = prediction.to('cpu')
        #     _, ind = torch.topk(prediction,5)

        #     for i in xrange(ind.shape[0]):
        #         label = labels[i]
        #         if label == ind[i][0]:
        #             classification_count += 1
        #             top5_count += 1
        #         elif label in ind[i]:
        #             top5_count += 1
        
        # print("Classification error is " + str(100 - classification_count/100.0) + "%")
        # print("Top 5 error is " + str(100 - top5_count/100.0) + "%")

        gc.collect()
        epoch += 1


print('Starting training')
run()
print('Training terminated')
