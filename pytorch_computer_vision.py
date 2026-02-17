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

#import pandas
import pandas as pd

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
epochs = 1

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


#TODO: make prediction and get model 0 results

torch.manual_seed(42)
def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn):
    """ returns a dictionary containing the results of model predicting on data_loader"""
    loss, acc = 0, 0 
    model.eval()
    with torch.inference_mode():
        ignostic = False
        if(next(model.parameters()).device.type == "cuda"):
            ignostic = True
        for X, y in tqdm(data_loader):
            if(ignostic == True):
                X, y = X.to(device), y.to(device)
            #make predictions
            y_pred = model(X)
            #accumlate the loss and acc values per batch

            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y,
                               y_pred=y_pred.argmax(dim=1))

        #scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)

    return {"Model_name": model.__class__.__name__, #only works when model is created with a class
            "Model_loss": loss.item(),
            "Model_acc": acc}

#calculate model 0 wretuls on test dataset

model_0_results = eval_model(model=model_0,
                           data_loader=test_dataloader,
                           loss_fn=loss_fn,
                           accuracy_fn=accuracy_fn)

print(model_0_results)


#TODO: set up device ignostic code

device = "cuda" if torch.cuda.is_available() else "cpu"

class FashionMNISTModelV1(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = input_shape, out_features = hidden_units),
            nn.ReLU(),
            nn.Linear(in_features = hidden_units, out_features = hidden_units),
            nn.ReLU(),
            nn.Linear(in_features = hidden_units, out_features =output_shape),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor):
        return self.layer_stack(x)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
model_1 = FashionMNISTModelV1(input_shape = 784,
                              hidden_units = 10,
                              output_shape = len(class_names)).to(device)
print(model_1, next(model_1.parameters()).device) #NOTE: all that next does is that it retrieves the next item in a iterable, which parameters return, so if we get the first item in the list of params, then we can check the device it is on, for which all of the params will be on the same device

#set up loss and optimizer

loss_fn = nn.CrossEntropyLoss()  #measure how wrong our model is 
optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr = 0.1) #tries to update our model's paremeteres to reduce the loss

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module,
               accuracy_fn,
               optimizer: torch.otpim,
               device: torch.device = device):
    """ trains the specifiied model """
    train_loss, train_acc = 0, 0 
    model.train()

    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y, y_pred = y_pred.argmax(dim=1))# logits to prediction labels

        optimizer.zero_grad()
            
        loss.backward()

        optimizer.step()

        if batch % 400 == 0:
            print(f"looked at {batch * len(X)} / {len(data_loader.dataset)}")

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    print(f"Training_loss: {train_loss:.4f}")
    print(f"Training_acc: {train_acc:.4f}")


train_step(model=model_1,
           data_loader=train_dataloader,
           loss_fn=loss_fn,
           accuracy_fn=accuracy_fn,
           optimizer=optimizer)

def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    """tests the specified model"""
    test_loss, test_acc = 0,0
    model.eval()
    with torch.inference_mode():
        for X,y in tqdm(data_loader):
            X,y = X.to(device), y.to(device)
            test_pred = model(X)
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true = y, y_pred = test_pred.argmax(dim=1))
        
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)

    print(f"\n Test_loss : {test_loss:.4f} | Test_acc : {test_acc:.4f}")


test_step(model=model_1,
          data_loader=test_dataloader,
          loss_fn = loss_fn,
          accuracy_fn = accuracy_fn,)


epochs = 1

start_time = timer() 

for epoch in range(epochs):
    print(f"Epoch: {epoch}")
    train_step(model=model_1,
           data_loader=train_dataloader,
           loss_fn=loss_fn,
           accuracy_fn=accuracy_fn,
           optimizer=optimizer)

    test_step(model=model_1,
          data_loader=test_dataloader,
          loss_fn = loss_fn,
          accuracy_fn = accuracy_fn,)

end_time = timer()

total_train_time_model_1 = print_train_time(start=start_time, end=end_time, device = next(model_1.parameters()).device)



#get model_1 results dictionary

model_1_results = eval_model(model = model_1,
                             data_loader = test_dataloader,
                             loss_fn = loss_fn,
                             accuracy_fn = accuracy_fn)


## TODO: Building a Convolutional Neural Network - CNNs are known for their capabilities to find patterns in visual data

#create a convolutional neural network
class FashionMNISTModelV2(nn.Module):
    """
        Model architecture that replicates the TinyVGG
        model from CNN explainer website.
    """
    def __init__(self, input_shape: int,
                       hidden_units: int,
                       output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels = input_shape,
                      out_channels = hidden_units,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_units,
                      out_channels = hidden_units,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels = hidden_units,
                      out_channels = hidden_units,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_units,
                      out_channels = hidden_units,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = hidden_units * 7 * 7, #there is a trick to calculate this, the two numbers at the end are the outputs of the second conv block
                      out_features = output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        #print(x.shape)
        x = self.conv_block_2(x)
        #print(x.shape)
        x = self.classifier(x)
        return x


torch.manual_seed(42)

model_2 = FashionMNISTModelV2(input_shape = 1,
                              hidden_units = 10,
                              output_shape = len(class_names)).to(device)

#TODO: breakding down the convolutional neural network 

#TODO: nn.Conv2d
# - 

torch.manual_seed(42)

#create a batch of images

images = torch.randn(size=(32, 3, 64, 64))
test_image = images[0]

print(f"Image batch shape: {images.shape}")
print(f"Single image shape: {test_image.shape}")
print(f"Test image: {test_image}")

#create a single Conv2d layer
conv_layer = nn.Conv2d(in_channels = 3, #in channels is the same number of color channels of the input image
                       out_channels = 10, #number of hidden units we have
                       kernel_size = 3, # this is equivalent to a tuple of n x n (3, 3) the kernel is the size of the group of data that the model with operate on
                       stride = 1, # how many pixels to jump over each convolution
                       padding = 0) # how many extra pixels to add on the edge incase there's important information on the edge

conv_output = conv_layer(test_image)
print(conv_output.shape)

#TODO: MaxPool2d  - just takes the most significant/biggest value within a certrain kernel size and so see if significant patterns persists if we simplfy the data further

print(f"Test image original shape: {test_image.shape}")

max_pool_layer = nn.MaxPool2d(kernel_size = 2)

test_image_through_conv = conv_layer(test_image)
print(f"shape after going through the conv_layer() {test_image_through_conv.shape}")

test_image_through_conv_and_max_pool = max_pool_layer(test_image_through_conv)
print(f"shape after going through conv and max pool layer {test_image_through_conv_and_max_pool.shape}")

##TODO: setting up a loss function and optimizer

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_2.parameters(),
                            lr = 0.1)
## training and testing our model_2 using our training and testing functions

torch.manual_seed(42)
torch.cuda.manual_seed(42)

#measure time
start_time = timer()

epochs = 3

for epoch in tqdm(range(epochs)):
    print(f"epoch: {epoch} \n-----")
    train_step(model=model_2,
               data_loader = train_dataloader,
               loss_fn = loss_fn,
               accuracy_fn = accuracy_fn,
               optimizer = optimizer,
               device = device)

    test_step(model = model_2,
              data_loader = test_dataloader,
              loss_fn = loss_fn,
              accuracy_fn = accuracy_fn,
              device = device)

end_time = timer()

total_train_time_model_2 = print_train_time(start = start_time, end = end_time, device = device)

#getting the model2 results

model_2_results = eval_model(model = model_2,
                             data_loader = test_dataloader,
                             loss_fn = loss_fn,
                             accuracy_fn = accuracy_fn)


compare_results = pd.DataFrame([model_0_results, model_1_results, model_2_results])
print(compare_results)

# add trainig time to results comparison
# performance speed trade off needs to be considered


compare_results["training_time"] = [total_train_time_model_0, total_train_time_model_1, total_train_time_model_2]
print(compare_results)

#visualize our model results

plt.figure()
compare_results.set_index("Model_name")["Model_acc"].plot(kind = "barh")
plt.xlabel("accuracy (%)")
plt.ylabel("model")
plt.savefig("model_comparison.png")

def make_predictions(model: torch.nn.Module,
                     data: list,
                     device: torch.device = device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            #prepare the sample
            sample = torch.unsqueeze(sample, dim=0).to(device)
            
            #forward pass (model outputs raw logits)
            pred_logit = model(sample)

            #get prediction probability (logit -> prediction probability)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)

            #get pred_prob off the gpu for further calculations
            pred_probs.append(pred_prob.cpu())


    #stack the pred_probs to turn list into a tensor
    return torch.stack(pred_probs)

import random
test_samples = []
test_labels = []

for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)

#view the first sample shape
test_samples[0].shape
plt.figure()
plt.imshow(test_samples[0].squeeze(), cmap="gray")
plt.title(class_names[test_labels[0]])
plt.savefig("MP.png")

# Make predictions
#
pred_probs = make_predictions(model=model_2,
                              data=test_samples)

#view the first two prediction probabilities
print(pred_probs[:2])
print(test_labels)

#convert prediction probabilities to labels
pred_classes = pred_probs.argmax(dim=1)
print(pred_classes)

# plot predictions
plt.figure(figsize=(9,9))
nrows = 3
ncols = 3
for i, sample in enumerate(test_samples):
    #create subplot
    plt.subplot(nrows, ncols, i + 1)

    #plot the target image
    plt.imshow(sample.squeeze(), cmap="gray")

    #find the prediction (in text form, e.g "sandal")
    pred_label = class_names[pred_classes[i]]

    #get the truth table (in test form)
    truth_label = class_names[test_labels[i]]

    #create a title for the plot
    title_text = f"Pred: {pred_label} | Truth: {truth_label}"

    #check for equality between pred and truth and change color of title text
    if pred_label == truth_label:
        plt.title(title_text, fontsize=10, c="g")
    else:
        plt.title(title_text, fontsize=10, c="r")

plt.savefig("predplt.png")


#making a confustion matrix for further prediction evaluation
#compares the predicted values with the true values in a tabular way, if 100% correct, all values in the matrix will be top left to bottom right (diagonal line)
# 1. make predictions with our trained model on the test dataset
# 2. make a confusion matrix 'torchmetrics.ConfusionMatrix'
# 3. plot the confusion matrix using 'mlxtend.plotting.plot_confusion_matrix()'
#

#import tqdm.auto
from tqdm.auto import tqdm

#1. make predictions with trained model
y_preds  = []
model_2.eval()
with torch.inference_mode():
    for X, y in tqdm(test_dataloader, desc="making predictions..."):
        X, y = X.to(device), y.to(device)

        y_logits = model_2(X)
        y_pred = torch.softmax(y_logits.squeeze(), dim=0).argmax(dim=1)
        y_preds.append(y_pred.cpu())

#concatenanate list of predictions into a tensor
print(y_preds)
y_pred_tensor = torch.cat(y_preds)
y_pred_tensor[:10]
print(y_pred_tensor)

import mlxtend
print(mlxtend.__version__)


from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

#2. setup confusion instance and copare predictions to targets
confmat = ConfusionMatrix(task="multiclass", num_classes=len(class_names))#now requires tasks
confmat_tensor = confmat(preds=y_pred_tensor,
                         target=test_data.targets)
print(confmat_tensor)

#3.plot the confusion matrix
fig, ax = plot_confusion_matrix(
    conf_mat = confmat_tensor.numpy(), #matplot lib likes working with numpy
    class_names = class_names,
    figsize=(10, 7)
)
plt.savefig("confusion_matrix.png")


##saving and reloading the trained model

from pathlib import Path

#create model dictory path

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True,
                 exist_ok=True)

#create model save
MODEL_NAME = "03_pytorch_computer_vision_model_2.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

#save the model state dict
print(f"Saving model tp : {MODEL_SAVE_PATH}")
torch.save(obj=model_2.state_dict(), # dont forget that the state dict is the model's learnt weight and baises
           f=MODEL_SAVE_PATH)

#create a new instance

torch.manual_seed(42)
loaded_model_2 = FashionMNISTModelV2(input_shape = 1,
                                     hidden_units = 10,
                                     output_shape = len(class_names))

#load in saved state dict()
loaded_model_2.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

#send model to targer device
loaded_model_2.to(device)

#evaluate loaded model

torch.manual_seed(42)

loaded_model_2_results = eval_model(
    model=loaded_model_2,
    data_loader = test_dataloader,
    loss_fn = loss_fn,
    accuracy_fn = accuracy_fn
)
print(loaded_model_2_results)
print(model_2_results)

#check if model results are close to each other
print(torch.isclose(torch.tensor(model_2_results["Model_loss"]),
              torch.tensor(loaded_model_2_results["Model_loss"]),
              atol=1e-02) )#this is adjusting tolerance levels

