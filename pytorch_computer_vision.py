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

plt.imshow(image.squeeze()) #this squeeze will remove that extra dimension
plt.title(label)
plt.imshow(image.squeeze(), cmap="gray")
plt.savefig("test_image.png")

torch.manual_seed(42)
torch.cuda.manual_seed(42)
fig = plt.figure(figsize=(9,9))
rows, cols = 4, 4
for i in range(1, rows*cols+1):
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    img, label = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(class_names[label])
    plt.axis(False)
    plt.savefig("test_image_collection.png")
    print(random_idx) 

from torch.utils.data import DataLoader

#batch size hyperparameter
BATCH_SIZE = 32

# turning datasets into iterables

train_dataloader = DataLoader(dataset=train_data, 
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False) #test data doesnt need to be shuffled because the model will only use it to verify rather than train off of it so the order doesnt matter


print(f"DataLoaders: {train_dataloader, test_dataloader}")
print(f"Length of the train_dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
print(f"Length of the test_dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")

#for batches that has remainders and is not exactly batch size of 32, the dataloader will handle it automatically so no need to worry about matching sizes of data and batch size

#check out wahts inside the training dataloader
train_features_batch, train_labels_batch = next(iter(train_dataloader))
print(train_features_batch.shape, train_labels_batch.shape)


torch.manual_seed(42)
random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
img, label = train_features_batch[random_idx], train_labels_batch[random_idx]
plt.figure()
plt.imshow(img.squeeze(), cmap="gray")
plt.title(class_names[label])
plt.axis(False)
print(f"Image size: {img.shape}")
print(f"Label : {label}, label_size: {label.shape}")
plt.savefig("dataloader.png")


# NOTE: Building a baseline model
# when starting to build a series of machine learning modelling experiement, its best practice to start with a baseline model.
#
# A baseline model is a simpel model you will try and improve upon with subsequent models/experiemtns.
#
# in other words, start with something simple and iterativly add complexity

#TODO: create a flatten layer

flatten_model = nn.Flatten()

#get a single sample

x = train_features_batch[0]

print(x, x.shape)

#Flatten the sample 
output = flatten_model(x) #perform forward pass

print(f"Shape before flattening : {x.shape}, Shape after flattening: {output.shape}")
#this results in one big vector of value
#NOTE: the flattening turn the shape from [color channels, height, width] ----> [color_channels, height*width]
