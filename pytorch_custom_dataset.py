#NOTE: we've used some datasets with pytorch before, but how do you get your own data into pytorch, use custom datasets

#domain libraries, depending on what you're workiing on , vision, text, audio, recommendation, you'll want to look into each of the PyTroch domain libraries for existing data loading functions and customizabel data loading functions.

import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

#our dataset is a subset of the food 101 dataset

#food 101 starts 101 different classes of food, our dataset starts with 3 classes of food and only 10% of the images (~75 training, 25 testing)

#why do this? 

#when starting out ML projects, it's important to try things on a small scale and then increase the scale when necessary. The whole point is to speed up how fast you can experiment


import requests
import zipfile
from pathlib import Path

#setup path to a data folder
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

#if the image folder doesn;t existm downlaod it and prepare it

if image_path.is_dir():
    print(f"{image_path} directory already exists... skipping download")

else:
    print(f"{image_path} does not exist, creating one ...")
    image_path.mkdir(parents=True, exist_ok = True)

#download the pizza, steak, sushi data

with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
    request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
    print("Downloading pizza, steak, sushi data...")
    f.write(request.content)

#unzip pizza, steak, sushi data

with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
    print("Unzipping pizza, steak, sushi data ..")
    zip_ref.extractall(image_path)

#becoming one with the data (data preparetion and data exploration)

import os

def walk_through_dir(dir_path):
    """Walks through dir_path returning its contents."""
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

#setup train and testing paths
train_dir = image_path / "train"
test_dir = image_path / "test"

print(train_dir, test_dir)

### 2.1 visualizing and imgae

#lets write code to :
#1. get all of the image paths
#2. pick a rnadom image path using python's random.choice()
#3. get the image class name using 'pathlib.Path.parent.stem
#4. Since we're working with images, let's open the image with Python's PIL, python pillow
#5. we'll then show the image and print metadata

import random
from PIL import Image


#set seed
random.seed(42)

#1. get all image paths
image_path_list = list(image_path.glob("*/*/*.jpg"))

print(image_path_list)

#2. pick a random image_path
random_image_path = random.choice(image_path_list)

print(random_image_path)

#3. get image class from path name (the image class is the name of the directory where the image is stored)
image_class = random_image_path.parent.stem
print(image_class)

#4. Open image 
img = Image.open(random_image_path) #note that if this is corrupt it may error

#5. Print metadata
print(f"Random image path: {random_image_path}")
print(f"Image class {image_class}")
print(f"Image height: {img.height}")
print(f"Image width: {img.width}")
print(img)


#visualizing the data using matplotlib instead

import numpy as np
import matplotlib.pyplot as plt

#turn the image into an array
img_as_array = np.asarray(img)

#plot the image with matplotlib
plt.figure(figsize = (10,7))
plt.imshow(img_as_array)
plt.title(f"Image class: {image_class} | Image_shape: {img_as_array.shape} -> [height, width, color_channels]")
plt.axis(False)
plt.savefig("image_pyplot.png")

#TODO: turning the images into tensors
#before we can use our image data with PyTorch
#1. turn target data into tensors (in our case, numerical representation of out images)
#2. turn it into a 'torch.utils.data.Dataset' and subsequently a 'torch.utils.data.DataLoader', we'll call these 'Dataset' and Dataloader

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#tranforming data with torchvision.transforms

#write a transform for image
data_transform = transforms.Compose([
    #resize  out image to 64x64
    transforms.Resize(size=(64, 64)),
    #flip the images randomly on the horzontal
    transforms.RandomHorizontalFlip(p=0.5),
    #turn the image into a torch tensor
    transforms.ToTensor()

])

print(data_transform(img))

#tranforming and visualizing the transformed images

#transforms help you get your images readdy to be used with a model/perform data augmentation

def plot_transformed_images(image_paths, transform, n=3, seed=None):
    """selects random images from a path of images and laods/trasnforms them then plots the original vs te transformed version"""

    if seed:
        random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    plt.figure()
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(nrows = 1,ncols = 2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original\nSize: {f.size}")
            ax[0].axis(False)

            #transform and plot target image
            transformed_image = transform(f).permute(1, 2, 0) # note we will need to change shape
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Tranforemd\nShape: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)

    plt.savefig("transformed.png")

plot_transformed_images(image_paths=image_path_list,
                        transform=data_transform,
                        n=3,
                        seed=42)

#NOTE: Option 1: loading image data using 'ImageFolder'

#we can load image classifictation data using torchvision.datasets.ImageFolder

#use ImageFolder to create dataset(s)
from torchvision import datasets
train_data = datasets.ImageFolder(root=train_dir,
                                  transform = data_transform, # transform for the data
                                  target_transform=None) #transform for the target/label

test_data = datasets.ImageFolder(root=test_dir,
                                 transform=data_transform)

print(train_data, test_data)

#get class names as a list
class_names = train_data.classes
print(class_names)

#get class names as a dict
class_dict = train_data.class_to_idx
print(class_dict)

# check the lengths of our dataset
print(len(train_data), len(test_data))
print(train_data.targets)

#index on the train_data Dataset to get a single image and label
print(train_data[0])
img, label = train_data[0][0], train_data[0][1]

print(img, label)

print(f"Image tensor:\n {img}")
print(f"Image shape: {img.shape}")
print(f"Image datatype: {img.dtype}")
print(f"Label: {label}")
print(f"Label datatype: {type(label)}")


#rearrange the order of dimensions
img_permute = img.permute(1, 2, 0)

#print out different shapes
print(f"Original: {img.shape} -> [color_channels, height, width]")
print(f"permuted: {img_permute.shape} -> [Heigh, width, color_channels]")

#plot the image
plt.figure(figsize=(10, 7))
plt.imshow(img_permute)
plt.axis("off")
plt.title(class_names[label], fontsize=14)
plt.savefig("impermuted.png")


## 4.1 turn loaded images into 'DataLoaders' 

# A 'DataLoader' is going to help us turn out 'Dataset' into iterables and we can customise the 'batch size'so out model can see 'batch_size' iamges at one time

#turn train and test datasets into datatloaders

import os
print(os.cpu_count())

from torch.utils.data import DataLoader
BATCH_SIZE =1
train_dataloader = DataLoader(dataset = train_data,
                              batch_size = BATCH_SIZE,
                              num_workers=0, #number of cpu cores used for this)
                              shuffle=True)
test_dataloader = DataLoader(dataset = test_data,
                              batch_size = BATCH_SIZE,
                              num_workers=0, #number of cpu cores used for this)
                              shuffle = False) #needs to be in the same order so dont shuffle)


print(train_dataloader, test_dataloader)
print(len(train_dataloader), len(test_dataloader))

img, label = next(iter(train_dataloader))

print(f"Image_shape: {img.shape} -> [batch_size, color_channels, height , width]")
print(f"Label shape: {label.shape}")

##NOTE: Option 2, loading image data with a custom 'dataset'
# 1. want to be able ot laod images from file
# 2. want to be able to get calss names from the dataset
# 3. want to be abe to get calsses as dictionary from the dataset

#NOTE: pros:
# can create a 'dataset' out of almost anything
# not limited to PyTorch pre-built 'dataset' functions
#
#NOTE: cons:
# even though you could create a 'dataset' out of almost anything, it doesnt mean it will work...
# using a custom dataset often results in us writing more code, which could be prone to errors or perfomance issues
#

import os
import pathlib
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple, Dict, List

#isntance of torchvision.datasets,ImageFolder()
print(train_data.classes, train_data.class_to_idx) #class to idx maps class folder names to unique integer indices

#NOTE: all pytorch datasets often subclass torch.utils.data.Dataset

#TODO: creating a helper function to get class names

#we want a function to :
#1. get the class names using 'os.scandir()' to traverse a target directory (ideally the directory is in standard image classification format)
#2. raise an error if the class anmes arent found (if this happens, there might be something wrong with the directory structure)
#3. turn the clas names into a dict and a lsit and return them

#setup path for target directory
target_directory = train_dir
print(f"target_dir: {target_directory}")

#get the class names from the target dirctory
class_names_found = sorted([entry.name for entry in list(os.scandir(target_directory))])
print(class_names_found)
print(list(os.scandir(target_directory)))

def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """ Finds the class folder names in a target directory."""

    #get the class names by scanning the target directory
    classes = sorted([entry.name for entry in list(os.scandir(target_directory)) if entry.is_dir()])
    
    #raise an error if class names could not be found
    if not classes:
        raise FileNotFoundError(f"couldnt find any classes in {directory} .. pls check file structure")

    #create a dictionary of index labels (computeres prefer numbers rather than strings as labels)
    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
    return classes, class_to_idx

print(find_classes(target_directory))


#TODO: create a custom 'dataset' to replicate 'ImageFolder'

# to create our own custom data set, we want to 
# 1. subclass 'torch.utils.data.Dataset'
# 2. init out subclass with a target directory (the directory we'd like to get data from) as well as a transform if we'd like t transform our data
# 3. create several attributes:
#  - paths - paths of our images
#  - transform - the transform we'd like to use
#  - classes - a list of the target classes
#  - class_to_idx - a fict of the target classes mapped to integer labels
# 4. create a functiojn to load images, this function will open an image
# 5. overwrite the '__len()__' method to return the length of our dataset
# 6. overwrite the '__getItem()__' method to return a given sample when passed an index

#write a custiom dataset class
from torch.utils.data import Dataset

#1. Subclass torch.utils.data.Dataset
class ImageFolderCustom(Dataset):
    #2. initialize our custom dataset
    def __init__(self,
                 targ_dir:str,
                 transform = None):
    
        #create class attirbutes
        #get all of the image paths
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg"))
        # setup transforms
        self.transform = transform
        #create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(targ_dir)

    #create a function to load images
    def load_image(self, index: int) -> Image.Image: #reminder that the arrow is jsut a type hint for the return type
        "Opens an image via a path and returns it"
        image_path = self.paths[index]
        return Image.open(image_path)

    #overwrite the __len__()
    def __len__(self) -> int:
        "returns the total number of samples"
        return len(self.paths)

    #overwrite the __getItem__()
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "returns one sample of data, data and lable (X, y)"
        img = self.load_image(index)
        class_name = self.paths[index].parent.name # this expects path in format: data_folder/class_name/image.jpg
        class_idx = self.class_to_idx[class_name]

        #transform is neccesary
        if self.transform:
            return self.transform(img), class_idx # return data, label (X, y)
        else:
            return img, class_idx # return untransformed image and label
