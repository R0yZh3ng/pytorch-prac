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
