#TODO: computer vision and convoluational neural networks

#NOTE: A convolutional neural network 


#import torch
import torch
from torch import nn

#import torchvision
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

#import matplot lib
import matplotlib.pyplot as plt

#check versions
print(torch.__version__)
print(torchvision.__version__)

#device ignostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

#TODO: Getting a dataset

#minist datasets are a large collection of datasets developed and used for early stage computer vision training

#the pytorch datasets library provides a lot of example datasets we can use download using torch

train_data = datasets.FashionMNIST(
    root = "data", #where to download data to
    train = True, #do we wan tht training_dataset
    download = True, #do we want to downlaod the dataset
    transform = torchvision.transforms.ToTensor(), # how do we want to transform the data
    target_transform = None # how do we want to tranform the labels/targets
)

test_data = datasets.FashionMNIST(
    root = "data", #where to download data to
    train = False, # setting it to false means that we are getting the testing dataset instead
    download = True, #do we want to downlaod the dataset
    transform = ToTensor(), # this also works and we dont need the entire path
    target_transform = None # how do we want to tranform the labels/targets
)

print(len(train_data), len(test_data))

image, label = train_data[0]
print(image, label)


class_names = train_data.classes
print(class_names)

class_to_idx = train_data.class_to_idx
print(class_to_idx)

print(f"Image shape: {image.shape} -> [color channel, height, width]") #NOTE: I believe that pytorch now supports color channel last for accerlerated cpu/gpu operations
print(f"Image lable: {class_names[label]}")

#theres only one color channel here because its grayscale images that we are using

#TODO: Visualizing our data as images

#plt.imshow(image) #this will result in a shape mismatch because imshow expects either just the heigth and width or color channels last, but since we are using grayscale images, we cant just squeeze the extra dimension

plt.imshow(image.squeeze())
plt.savefig("test_image.png")

