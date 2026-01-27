import torch
from torch import nn
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd

#NOTE: Covering
# - architesture of  a neural network classification model
# - input and output shapes of a classidication model(features and labels)
# - creating custom data to view, fit on and predict on
# - steps in modelling
#  - creating a model, setting a loss function and optimiser, creating a training loop, evaluating a model
# - saving and loading models
# - harnessing the power of non-linearity
# - different classidication evaluation methods


#numerically represent the input images as 224w 224h and color channle 3, then produce the probability that the input image is a certain class

#batch size is the number of inputs to process each time, 32 is a common number to use, it is efficient to use batch size in the multiples of 8

#NOTE: high level architectiuure of a classificiation model
# - input layer shape(in_features)
# - hidden layers
# - neurons per hidden layer
# - output layer shape (out_features)
# - hidden layer activation
# - output activation
# - loss function
# - optimizer

#problem of prediciton whether something is one or another (can have multiple output options)

import sklearn #popular ml library
from sklearn.datasets import make_circles # make circles generate a synthetic 2d dataset . creates points arrangeed into two concerntric circles, each assigned a different class lable of 0 or 1, bascially its a binary classification toy dataset where depending on the value of x1 and x2 you determine whether its a inside circle or a outside circle, and the 0 and 1 is the label y

n_samples = 1000

X, y = make_circles(n_samples, noise=0.03, random_state = 42)

print(f"first 5 samples of X: {X[:5]}")
print(f"first 5 samples of y: {y[:5]}")

#make a dataframe of a circle data

circles = pd.DataFrame({"X1" : X[:, 0],
                        "X2" : X[:, 1],
                        "label" : y})

print(circles.head(10))

plt.figure()
plt.scatter(x = X[:, 0],
            y = X[:, 1],
            c = y,
            cmap=plt.cm.RdYlBu)

plt.savefig("testCircle.png")

#note that the data we're working with is often referred to as a toy dataset, a dataset that is small enough to experiment but still sizeable enough to practice the fundamentals

#TODO: 1.1 input and output shapes

print(X.shape)
print(y.shape)

#view the frist exmaple of features and labels
X_sample = X[0]
y_sample = y[0]

print(f"Values for one sample of X: {X_sample}, shape of X: {X_sample.shape}")
print(f"Values for one sample of y: {y_sample}, shape of y: {y_sample.shape}")

#this is to help me understand what the input and output shapes should look like so i dont encounter shape mismatch errors

#TODO: 1.2 turn data into tensors and create train and test splits

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

#split data into training and test sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2, #0.2 means that 20% of data will be test and 80% will be train,
                                                    random_state=42)

#TODO: building a model

#lets build a model to classify our red and blue does

#to do so, we need to
# 1. setup device agonistic code to out code will run on a gpu if available
# 2. construct a mode (sub subclassing 'nn.module')
# 3. define a loss function and optimizer
# 4. create a training and test loop
#

device = "cuda" if torch.cuda.is_available() else "cpu"

X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

print(X_train.device)
print(torch.cuda.is_available())

# we we ve setup device agnostic code, lefts create a model that
#  - subclass nn.Module (almost all models in PyTorch subclass nn.Module)
#  - create 2 nn.linear laters that are capable of handling the shapes of our data
#  - define a forward method that outlines the forward pass of the model
#  - instantiatte an instance of our model class and sent it to the target device
#

#TODO: 1. construct a model that subcalsses nn.Module

class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        #2. create 2 nn.linear laters capable of handling the shapes of our data
        self.layer_1 = nn.Linear(in_features=2, out_features=5) #layer of out for the previous layer must match layer of input for the next layer
        self.layer_2 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        return self.layer_2(self.layer_1(x)) #x -> layer_1 ->layer_2 ->output

#4. instatiate an instance of our model class and sent it to the target device

model_0 = CircleModelV0().to(device)
# print(model_0.device) - this doesn work 
# do this instead
print(next(model_0.parameters()).device)

#NOTE: can visit tensorflow playground to visualize the neural network

#lets replicate the model above using nn.Sequential():

model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
).to(device)

# this here basically implements the same thing, like the names suggest its jsut going through the specified layers in sequence
#NOTE: useful and can directly implement into the class by initializing this in the init of the class and calling it through the forward function

#make some predictions with the model
with torch.inference_mode():
    untrained_preds = model_0(X_test)
print(f"First 10 predictions: {untrained_preds[:10]}")
print(f"First 10 labels: {y_test[:10]}")
print(untrained_preds[:10]==y_test[:10])

#set up the loss function
# loss_fun = nn.BCELoss() # this works but needs to have go through the sigmoid activation function already
loss_fn = nn.BCEWithLogitsLoss() # this has the sigmoid activation function built in which makes the math more stablea

optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)

#calculate accuracy of prediction out of 100
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc

# train model
#
# we need to build a training loop and testing loop
# our model outputs are going to be raw logits
# we can convert these logits into prediciton probabilites by passihng them to some kind of activation function (sigmoid for binary cross entropy and softamx for multiclass classficiation)
# then we can convert our models prediction probabilities to prediction labels by either rondung them or taking the argmax() (this would be for multiclass)
#
#view the first 5 outputs of theh forward pass on the test data
model_0.eval()
with torch.inference_mode(): 
    y_logits = model_0(X_test)[:5]
print(y_logits)

y_pred_probs = torch.sigmoid(y_logits) # round only after the sigmoid because the weigths and bias need to be first turned into prediction  probabiliies 
y_pred = torch.round(y_pred_probs)
y_pred_label = y_train[:5]
print(torch.eq(y_pred.squeeze(), y_pred_label.squeeze()))  #the squeeze just gets rid of the extra dimension

#TODO: building a training and test loop
torch.manual_seed(42)
torch.cuda.manual_seed(42) #this also works if ur working on a cuda device


epochs = 100

X_train, y_train = X_train.to(device), y_train.to(device) #making sure data is on the right device
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):
    ### Training
    model_0.train()

    #1. forward pass 
    y_logits = model_0(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits)) #turn logits into pred probs then into pred lables, sigmoid is a actiation fucntion

    #2. Calculate loss/accurarcy

    loss = loss_fn(y_logits, y_train) #nn.BCEWithLogitsLoss, expect raw logits as input where as if we only use BCELoss then we need to convert and use the sigmoid first

    acc = accuracy_fn(y_true = y_train, y_pred = y_pred)


    #3. optimizer zero grad

    optimizer.zero_grad()

    #4. loss backward
    loss.backward()

    #5. optimzer step
    optimizer.step()


    ### Testing
    model_0.eval()
    with torch.inference_mode():
        #forward pass
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        
        #2. calculate test loss/acc
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true = y_test, y_pred = test_pred)

    #print out whats happening
    if epoch % 10 == 0:
       print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f} | Test loss: {test_loss:5f}, Test acc: {test_acc:.2f}%")
    
##NOTE: make predictions and evaluate the model

# from the metrics it looks like our model isnt learning anything..
#
# so to insepct it lets make some predictions and make them visual
#
# Visualization is key in understanding whether the model is effective
#
# to do so, we're going to import a function called plot_decision_boundry() 

import requests
from pathlib import Path

#download helper functions from learn PyTorch repo (if its not already downloaded)
#dont forget that the link needs to be raw

if Path("helper_function.py").is_file():
    print("helper_functions.py already exists, skipping download")
else:
    print("Downloading helper_functions.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary

#plot decision boundry of the model

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_0, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_0, X_test, y_test)
plt.savefig("plot_decision_boundary_intro.png")



#TODO: improving our model(from a model perspective)

#add more layers (giving the model more chances to learn about patterns in the data)
#add more hidden units (go from 5 hidden units to 10 hidden units) (hidden units transform raw input into abstract, high level feature representations, allowing the model to learn complex, non linear patterns)
#fit for longer (more epochs so more chances to adjust, but dont forget to adjust learning rate accordingly if wanting to get more accurate)
#changing the activation fucntions
#changing the learning rate (bigger/smaller steps allows for more accurate adjustments rather than encountering either a vanaishing or exploding gradient problem)

# these options are all from a model's perspective becayse they deal directly with the model, rather than the data
# and because these options are all values we (as machine learning engineers and data scientist can change, they are referred to as ) "hyperparameters"

class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1=nn.Linear(in_features=2, out_features=10)
        self.layer_2=nn.Linear(in_features=10, out_features=10)
        self.layer_3=nn.Linear(in_features=10, out_features=1)

    def forward(self, x):
        #z = self.layer_1(x) #NOTE: usually z is used to represent logits
        #z = self.layer_2(z)
        #z = self.layer_3(z)
        #return z # we can use the all in one way of writing it to take advantage of whatever speedup we can get
        return self.layer_3(self.layer_2(self.layer_1(x)))


model_1 = CircleModelV1().to(device)


#Create a loss function

loss_fn = nn.BCEWithLogitsLoss() # this has the sigmoid activation function built in which makes the math more stablea
#Create a optimizer
optimizer = torch.optim.Adam(params=model_1.parameters(),
                            lr=0.1)

#write a training and evaluation loop for model 1
torch.manual_seed(42)
torch.cuda.manual_seed(42)


#train for longer

epochs = 1000

#put data on the target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):
    model_1.train()
    
    #forward pass
    y_logits = model_1(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    #calculate the loos
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_pred)
    #optimizer zero grad
    optimizer.zero_grad()

    # loss backward
    loss.backward()
    

    #optimizer step
    optimizer.step()

    ##testing

    model_1.eval()
    with torch.inference_mode():
        #1. forward pass
        test_logits = model_1(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        #caclualte the loss
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)

    #print out whats happening
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc : {acc:.2f},Test Loss: {test_loss:.5f}, Test acc: {test_acc:.2f}")


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_0, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_0, X_test, y_test)
plt.savefig("plot_decision_boundary_modelV1.png")



###TODO: preparing data to see if out model can fit a straight line

#one way to troubleshoot a larger problem is to test out smaller problems



#create some data 
weight = 0.7
bias = 0.3
start = 0 
end = 1
step = 0.01

#create data
X_regression = torch.arange(start, end, step).unsqueeze(dim=1)
y_regression = weight*X_regression + bias #linear regression formula (without epsilon)

print(len(X_regression))
print(X_regression[:5], y_regression[:5])

#create train and test splits

train_split = int(0.8 * len(X_regression))
X_train_regression, y_train_regression = X_regression[:train_split], y_regression[:train_split]
X_test_regression, y_test_regression = X_regression[train_split:], y_regression[train_split:]

plot_predictions(train_data = X_train_regression,
                 train_labels = y_train_regression,
                 test_data = X_test_regression,
                 test_labels = y_test_regression)


#same architecture as model_a (but using nn.sequential)

model_2 = nn.Sequential(
    nn.Linear(in_features=1, out_features=10),
    nn.Linear(in_features=10, out_features=10),
    nn.Linear(in_features=10, out_features=1) 
).to(device) #this is the same as model one except change the number of in features to make sure that it matches a linear setup


loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params = model_2.parameters(), lr=0.1)

#train the model

torch.manual_seed(42)
torch.cuda.manual_seed(42)


epochs = 1000

X_train_regression, y_train_regression = X_train_regression.to(device), y_train_regression.to(device)
X_test_regression, y_test_regression = X_test_regression.to(device), y_test_regression.to(device)

for epoch in range (epochs):
    y_pred = model_2(X_train_regression)
    loss = loss_fn(y_pred, y_train_regression)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_2.eval()
    with torch.inference_mode():
        test_pred = model_2(X_test_regression)
        test_loss = loss_fn(test_pred, y_test_regression)


    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss: 5f} | Test loss: {test_loss: .5f}")

model_2.eval()
with torch.inference_mode():
    y_pred = model_2(X_test_regression)

plot_predictions(train_data = X_train_regression.cpu(),
                 train_labels = y_train_regression.cpu(),
                 test_data = X_test_regression.cpu(),
                 test_labels = y_test_regression.cpu(),
                 predictions=y_pred.cpu())


#TODO: The missing piece, nonlinearity

#recreating non-linear data
#make and plot data

n_samples = 1000

X, y = make_circles(n_samples,
                    noise = 0.03,
                    random_state=42)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)

#convert data to tensors and then to train and test splits
from sklearn.model_selection import train_test_split

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



#TODO: building a model with non-linearity

class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU() #relu bascially takes all values < 0 and make it 0, and keeps all values > 0 as it is

    def forward(self, x):
        #where should we put our non-linear activation functions? 
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x))))) #relu should go through every layer, so if using nn.Sequential you should put a relu between every layer


model_3 = CircleModelV2().to(device)
print(model_3)

#artificial neural networks are just a collection of linear and non linear function which are potentially able ot find patterns in data

#setup loss and optimizer

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model_3.parameters(), lr = 0.01)

#training a model with non-linearity

#random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

#put all data on the target device

X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

epochs = 10000

for epoch in range(epochs):
    model_3.train()

    y_logits = model_3(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true = y_train,
                      y_pred = y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_3.eval()
    with torch.inference_mode():
        test_logits = model_3(X_test).squeeze() # this squeeze is very important in order to make the shapes match
        test_pred =  torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true = y_test,
                          y_pred = test_pred)

    if epoch % 1000 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.4f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% ") #the .2f just says to the specified decimal places

#make predictions

model_3.eval()
with torch.inference_mode():
    y_preds = torch.round(torch.sigmoid(model_3(X_test).squeeze()))



plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_3, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_3, X_test, y_test)
plt.savefig("plot_decision_boundary_modelV3.png")


