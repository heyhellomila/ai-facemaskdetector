# -*- coding: utf-8 -*-
"""cnn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pAicnuXZaUELSg0IZEckX_FzNEzyCoFY
"""
from statistics import fmean

import torch

torch.cuda.is_available()

# imports
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

#from google.colab import drive
#drive.mount('/content/drive')

#Chosing which device to run the training on. Uncomment the one you need.
#device = 'cpu'
device=torch.device("cuda")



#The flag is to save based on accuracy or save based on loss
based_on_loss = True
#The flag triggers testing-phase-only mode. (loading a pretrained model)
test_only = False


import gc
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

gc.collect()

# Clearing Cuda cache
torch.cuda.empty_cache()

def get_data():
    #data_dir = '/content/drive/MyDrive/classified'
    data_dir = 'C:/Users/bl/Documents/CNNcomp472/classified'

    data_dir_male = 'C:/Users/bl/Documents/CNNcomp472/gender-20220622T010754Z-001/gender/male'

    
    
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((256,256)),
                                    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]) 

    
    original_set = datasets.ImageFolder(root=data_dir, transform=transform)  # dataset

    male_set = datasets.ImageFolder(root=data_dir_male, transform=transform)

    print(original_set.classes)
    n = len(original_set)  # total number of examples
    n_test= int(0.25*n)# take ~25% for test
    for x in range(0, 9): # rounding it to be divisible by 4
      n_test+=1
      if n_test % 4 == 0: 
        break 

    # test_set = torch.utils.data.Subset(original_set, range(n_test))  # take first 25%
    # train_set = torch.utils.data.Subset(original_set, range(n_test, n))  # take the rest


    train_set, test_set = torch.utils.data.random_split(original_set, [n-n_test, n_test])



    train = DataLoader(train_set, batch_size=32, shuffle=True)
    #*******test = DataLoader(test_set, batch_size=int(len(test_set)/4), shuffle=False)
    test = DataLoader(test_set, batch_size=1000, shuffle=False)

    #test = DataLoader(male_set, batch_size=1000, shuffle=False)

    print(len(train_set))
    print(len(test_set))
    print(n)
    

    return train, test, train_set, test_set, original_set

train, test, train_set, test_set, original_set = get_data()

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.1),
        )

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.1),
        )

                
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.1),
        )
        
        self.fc_layer1 = nn.Sequential(
            nn.Linear(32 * 32 * 128, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
        )

        self.fc_layer2 = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
        )

        self.fc_layerFinal = nn.Sequential(
            nn.Linear(1000, 4),
        )
        
    def forward(self, x):
        # conv layers
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layers
        x = self.fc_layer1(x)
        x = self.fc_layer2(x)
        x = self.fc_layerFinal(x)
        return x

num_epochs = 25
num_classes = 4
learning_rate = 0.001


if test_only == False :
    #************************
    # Setting up the model
    model = CNN()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_step = len(train)
    loss_list = []
    acc_list = []

    iterations=[]
    iterations_improved=[]#graph
    epoch_iterations=[]

    accuracyListPerEpoch = [] # Keeps track of the accuracies for each epoch
    accuracyListPerEpoch_improved = []

    lossListPerEpoch = []
    lossListPerEpoch_improved = []

    # The training. Only saving the model, after each epoch/iteration (depending on flag) , if the accuracy improves. Printing accuracies at each epoch.
    n=0 # For the graph
    n2=0 # For the epoch graph

    if based_on_loss == False:
        print("saving/rejecting the following epoch based on accuracy:")
    if based_on_loss == True:
        print("saving/rejecting the following epoch based on loss:")

    for epoch in range(num_epochs):

        message = "rejected"

        correct_single_epoch = 0.0
        loss_single_epoch = 0.0

        if os.path.exists('C:/Users/bl/Desktop/cnnModelSaves/model.pth'):
            model.load_state_dict(torch.load('C:/Users/bl/Desktop/cnnModelSaves/model.pth'))
            model = model.to(device)

        for i, (images, labels) in enumerate(train):
            # Move them to device (cuda or cpu)
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())
            # Backprop and optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Train accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            correct_single_epoch += (predicted == labels).sum().item()
            loss_single_epoch += loss.item()

            #For the graph
            iterations.append(n)
            n+=1


        #Claculating the accuracy of this epoch and adding it to the array
        accuracy_this_epoch= (correct_single_epoch / len(train.dataset)) * 100
        loss_this_epoch = loss_single_epoch / n

        accuracyListPerEpoch.append(accuracy_this_epoch)
        lossListPerEpoch.append(loss_this_epoch)

        if based_on_loss == False:
            # Checking if the accuracy has improved this epoch. If so it will save that model,
            # otherwise it will just reload the previous  model and continue from there.
            if ((epoch <= 0) or (accuracyListPerEpoch[epoch] >= accuracyListPerEpoch_improved[-1])):
                torch.save(model.state_dict(), 'C:/Users/bl/Desktop/cnnModelSaves/model.pth')

                # For the graph
                epoch_iterations.append(n2)
                n2 += 1
                accuracyListPerEpoch_improved.append(accuracyListPerEpoch[epoch])

                message = "saved"

        if based_on_loss == True:
            # Checking if the loss has improved this epoch. If so it will save that model,
            # otherwise it will just reload the previous  model and continue from there.
            if ((epoch <= 0) or (lossListPerEpoch[epoch] <= lossListPerEpoch_improved[-1])):
                torch.save(model.state_dict(), 'C:/Users/bl/Desktop/cnnModelSaves/model.pth')

                # For the graph
                epoch_iterations.append(n2)
                n2 += 1
                lossListPerEpoch_improved.append(lossListPerEpoch[epoch])

                message = "saved"

                #graph
                accuracyListPerEpoch_improved.append(accuracyListPerEpoch[epoch])

        print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%, Status: {}'
            # print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
            .format(epoch + 1, num_epochs, loss_this_epoch,
            accuracy_this_epoch, message))



    ######################################
    # Plotting
    # The code for the plotting is taken from https://www.cs.toronto.edu/~lczhang/360/lec/w02/training.html
    plt.plot(iterations, loss_list)
    plt.title("Training Curve (batch_size={}, lr={})".format(len(train_set), learning_rate))
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()
    plt.plot(iterations, acc_list)
    plt.title("Training Curve (batch_size={}, lr={})".format(len(train_set), learning_rate))
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy")
    plt.show()

    plt.plot(epoch_iterations, accuracyListPerEpoch_improved)
    plt.title("Training Curve per epoch improved only (batch_size={}, lr={})".format(len(train_set), learning_rate))
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy")
    plt.show()
    ######################################




    # Saving the trained model and emptying the GPU memory
    torch.save(model.state_dict(), 'C:/Users/bl/Desktop/cnnModelSaves/model.pth')
    #del model
    torch.cuda.empty_cache()

    #************************************


#Loading the model before testing
device = 'cpu'
model = CNN()

#if os.path.exists('C:/Users/bl/Desktop/cnnModelSaves/model.pth'):
model.load_state_dict(torch.load('C:/Users/bl/Desktop/cnnModelSaves/model.pth'))
#else:
    #print("****The folder is empty*****")

model = model.to(device)

# Overall accuracy
with torch.no_grad():
    correct = 0
    total = 0
    for pred_images, pred_labels in test:
        pred_images = pred_images.to(device)
        pred_labels = pred_labels.to(device)
        pred_outputs = model(pred_images)
        _, pred_predicted = torch.max(pred_outputs.data, 1)
        total += pred_labels.size(0)
        correct += (pred_predicted == pred_labels).sum().item()

        accuracy= (correct / len(test.dataset)) * 100

    #(correct / total) * 100
    print('Accuracy of the network on the {} test images: {} %'.format(len(test.dataset), accuracy))

# print tensors
classes = original_set.classes
with torch.no_grad():
    model.eval()
    correct = 0
    total = 0
    for images, labels in test:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        print("Predicted: ",predicted, len(predicted))
        print("Expected: ",labels,len(labels))
        total += labels.size(0)

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report

eval_labels = pred_labels.detach().cpu()
eval_predictions = pred_predicted.detach().cpu()


print(classification_report(eval_labels, eval_predictions))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib.pyplot import figure

figure(figsize=(80, 60), dpi=80)
cm = confusion_matrix(eval_labels, eval_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=original_set.classes)
disp.plot()

plt.show()