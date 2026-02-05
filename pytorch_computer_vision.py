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

class FashionMNISTModelV0(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = input_shape,
                      out_features = hidden_units),
            nn.Linear(in_features = hidden_units,
                      out_features = output_shape),
        )

    def forward(self, x):
        return self.layer_stack(x)


torch.manual_seed(42)
torch.cuda.manual_seed(42)

#set up model with input parameters

model_0 = FashionMNISTModelV0(input_shape = 784, #this is 28 x 28
                              hidden_units = 10, #how many units in the hidden layer
                              output_shape = len(class_names)).to("cpu") # one for every class).

print(model_0)

dummy_x = torch.rand([1, 1, 28, 28])
print(model_0(dummy_x).shape) # the flatten layer combins the image into one tensor with a value for each class

## setting up loss, optimizer and evalutation metrics

# - loss function - since we were working with multi class data, out loss function will be cross entropy loss
# - optimizer  - stochastic gradient descent
# - evaluation metric - since its a classification problem, lets use accuracy as our metric


from helper_functions import accuracy_fn

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params = model_0.parameters(),
                            lr = 0.1)

#machine learning is very experimental, two of the main things you'll often want to track are:
# - Model's performance (loss, accuracy values, etc)
# - How fast it runs


from timeit import default_timer as timer 
def print_train_time(start: float,
                     end: float,
                     device: torch.device = None):
    """ prints difference between start and end time. """
    total_time = end - start
    print(f"train time on {device}: {total_time:.3f} seconds")
    return total_time

start_time = timer()
#some code
end_time = timer()
print_train_time(start=start_time, end=end_time, device="cpu")


### NOTE: Creating a training loop and training a model on batches of data


# 1. loop through epochs
# 2. loop through training batches, perfomr training steps, calcualte the training loss *per batch*
# 3. loop through testing batches, perform testijng stpes, calculate the test loss *per batch*
# 4. print out whats happeninh
# 5. time it all 


from tqdm import tqdm

torch.manual_seed(42)
train_time_start_on_cpu = timer()

#set the number of epochs (we'll keep this all for faster training times)
epochs = 3

#creating training and test loop
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n------")

    ###training 
    train_loss = 0
    # add a loop to loop through the training batches
    for batch, (X, y) in enumerate(train_dataloader):
        model_0.train()

        y_pred = model_0(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss #accumulate the loss within the batch and divide afterwards to get a averge loss of the batch

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #print out what happening

        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples.")
    
    train_loss /= len(train_dataloader)

    test_loss, test_acc = 0, 0
    model_0.eval()
    with torch.inference_mode():
        for X_test, y_test in test_dataloader:
            test_pred = model_0(X_test)
            test_loss += loss_fn(test_pred, y_test)
            test_acc += accuracy_fn(y_true=y_test, y_pred = test_pred.argmax(dim = 1))

        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)

    print(f"\n Train_loss : {train_loss:.4f} | Test_loss : {test_loss:.4f} | Test_acc : {test_acc:.4f}")

train_time_end_on_cpu = timer()
total_train_time_model_0 = print_train_time(start = train_time_start_on_cpu, end = train_time_end_on_cpu, device = str(next(model_0.parameters()).device))
#next(model_0.parameters).device gets the device

