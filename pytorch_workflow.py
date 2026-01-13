#NOTE: Pytorch workflow
#getting data ready
#building or training a pretrained model to suit the problem - 1. picking a loss function and optimizer -> build a training loop
#fit the model to the data and make a prediction
#evaluate the model
#improve through experiementation
#save and reload the trained model


#TODO: 1. data processing and loading
#TODO: 2. building the model
#TODO: 3. fitting the model to data (training)
#TODO: 4. making predictions and evaluting a model (inference)
#TODO: 5. saving and loading a model
#TODO: 6. putting it all together

import torch
from torch import nn #NOTE: nn contains all of pytorch's building blocks for neural networks
import matplotlib.pyplot as plt
import numpy as np

#check PyTorch version
print(torch.__version__)

#TODO: DATA PREP AND LOADING

#machine learning is a game of two parts, gettijng data into a numerical representation, build a model to learn patterns in that representation

#to showcase this, lets create some known data using the linear regression formula

#create known parameters
weight = 0.7
bias = 0.3

#create
start = 0
end = 1
step = 0.02
x = torch.arange(start, end, step).unsqueeze(dim = 1)
y = weight * x + bias

print(x[:10], y[:10], len(x), len(y))

#splitting the data into training set(60-80%), validation set(10-20%) and test set(10-20%)

#NOTE: creatinga train/test split
#x and y are effectively the training data and and the data labels because y is a function of x
#
train_split = int(0.8 * len(x))
x_train = x[:train_split]
y_train = y[:train_split]
x_test = x[train_split:]
y_test = y[train_split:]

print(len(x_train), len(y_train), len(x_test), len(y_test))

#TODO: how might we better visualize our data?

def plot_predictions(train_data = x_train, train_label = y_train, test_data = x_test, test_label = y_test, predictions = None):

    plt.figure(figsize = (10, 7))

    plt.scatter(train_data, train_label, c = "b", s=4, label="Training Data") #this function creates scatter plots 
 
    plt.scatter(test_data, test_label, c = "g", s=4, label="Testing Data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c = "r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})
    #plt.show() #NOTE: note that plt.show() needs to be ran at the end in local files, but since using terminal there's no gui component to display the image, instead save it and load it manually
    plt.savefig("testLG.png")
 
plot_predictions()


#TODO: building the first pytorch model


#create a linear regression model class

class LinearRegressionModel(nn.Module): #NOTE: almost everything in PyTorch inherits off nn.modules, this is the base class for all nerual network modules > YOUR MODLES SHOULD ALSO SUBCLASSS THIS CLASS!!!, MODULES CAN ALSO CONTAIN OTHER MODULES
    def __init__ (self):
        super().__init__()

        #NOTE: initialize model parameters to be used in various computations(these could be different layers from torch.nn, single parameters, hard-coded values or functions)
        self.weights = nn.Parameter(torch.randn(1, requires_grad = True, dtype=torch.float)) 
        #parameter is a kind of tensor that is to be considere a module parameter. they are tensor subclasses that have a special property when used with module, when they're assigned as module attributes they are automatcially added ot the list of its parameters and will appear in the module.parameters iteratos.
        #required_grad - does the parameter require gradient
        self.bias = nn.Parameter(torch.randn(1, requires_grad = True, dtype=torch.float))

        # Forward method to define the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor: # <- "x" is the input data #NOTE: x: torch.Tensor this is a type hint to make sure that the input arguments matches the expected
        return self.weights * x + self.bias # this is the linea regression formula

# this is just a basic gradient desent model, where the weigth and biases are adjusted iteratively by the forward function in the model until it matches the data
#
# the model first starts with random values for the weigth and bias, 
# looks at the training data and adjust the random values to better represent the ideal values
#
# the model does this througjh
#  - gradient desent            - this is why we set the requires_grad to to be true because we need the partial derivatives of the weight and bias comapred to the data to continously decrease until it reaches zero or hits our epoch limit
#  - back propagation

#NOTE: what is the forward function?
#forward is what defines the the forward computation of the the model at every call, any subclass of nn.Module needs to override forward()

#NOTE: Pytorch model building essentials
# - torch.nn - contains all of the building blocks for computational graphs (a neural network can be considerd a computational graph)
# - torch.nn.Parameter - what parameters should our model try and learn, often a PyTorch layer from torch.nn will set these for us
# - torch.nn.Module - the base class for all neural network modules, if you subclass it, you should overwrite forward()
# - torch.optim - this where the optimzers in PyToch live, they will help with gradient descent

### checking the contents of our PyTorch model

# Create a random seed
torch.manual_seed(42)

#create an instance of the model (this is a subclass of nn.module)
model_0 = LinearRegressionModel()

print(list(model_0.parameters()))

#list named parameters
print(model_0.state_dict())

##making predictions using 'torch.inference_mode('

#to check our model's predictive power, lets see how well it predicts y_test based on x_test
# when we pass data through our model, its going to run it through the forward() method
#
# make prediction with model


with torch.inference_mode(): #inference mode just disables the in tracking that the parameter would normally have like the requires_grad, this gets rid of the unnecessary overhead when only testing the predictions
    y_preds = model_0(x_test)

print(y_preds)

plot_predictions(predictions=y_preds)

#NOTE: train model
# the whole idea of training is for a model to move from some unknown parameters (these may be random) to some known parameters
# # or in other words from a poor representation of the data to a better representation of the data
#

#NOTE: one way to measure how poor or how wrong your models predictions are is to use a loss function, can also be refered to as cost function or criterion
# difference comapreed to expected output, lower is better
#
#NOTE: optimizer:
# takes into account the loss of a model and adjusts the mode's prarmeters to improve the loss function
#

#TODO: we need
# 1. a training loop
# 2. a testing loop
#


#set up a loss function
loss_fn = nn.L1Loss() #L1 loss is just the absolute error between the expected value and the input value

#set up a optimizer 
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.0001) #stochastic gradient descent, perform gradient descent by randomly adjusting values and following a direction that lowers the loss function
                                                          #lr (learnig rate) is possibily the most important hyperparameter you can set, just how big of a step in the param you can take
### building a training loop

# 0. loop through the data
# 1. forward pass (this involves data moving through our model"s 'forward()' functions - also called forward propagation 
# 2. calculate the loss (conpare forward pass predictions to ground truth lables)
# 3. opitimizer zero grad
# 4. loss backward - move backwards through the network to calculate the gradients of each of the parameters of out model with respect to the loss (back propagation)
# 5. optimizer step 0- use the optimizer to adjust outr model's parametrs to improve the loss. (gradient descent)
#
#NOTE: an epoch is one loop through the data (this is a hyperparameter because we set it ourselves)
epochs = 18000
#NOTE: if decreasing the learning rate, then the number of epochs need to be increased to account for the increased number of steps requireed to reach the goal
##training


#track different values so you can compare past and future experiments
epoch_count = []
loss_values = []
test_loss_values = []

#0. loop through the data / step through the model for a number of epochs
for epoch in range(epochs):

    model_0.train() #train mode in PyTorch set all parameters taht require gradients to require gradients
    
    #1. forward pass
    # calls the forward method defined in the model declaration
    y_pred = model_0(x_train)

    #2. calculate the loss / how diffrent are the models predictions to the true values
    loss = loss_fn(y_pred, y_train) #(input, target)

    #3. optimizer zero grad/ zero the gradients of the optimizer
    optimizer.zero_grad()

    #4. perform backpropagation on the loss with respect to the parameteres of the model/ compute the gradient of every parameter with requires grad = true
    loss.backward()

    #5. step the optimizer (perform gradient descent) / update teh model's paremeters with respect to the gradients calculated by loss.backward()
    optimizer.step() #NOTE: by default, how the optimizer changes will accumulate through the loop, so we have to zero them above in step three for the next iteration of the loop


    #tell the model we're evaluating now
    model_0.eval() #turns off different settings in the model not needed for evaluation/testing (dropout/batch norm layers)

    with torch.inference_mode(): # turns off gradient tracking
        #1. Do the forward pass in the 
        test_pred = model_0(x_test)

        #2. calculate the loss
        test_loss = loss_fn(test_pred, y_test)
    
    if epoch % 1000 == 0: #dont gotta print every epoch, every 10 or any other intervals are equally valid
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)
        print(f"Epoch: {epoch} | Loss: {loss} | Test_loss: {test_loss}")
        print(model_0.state_dict())


 

with torch.inference_mode():
    y_preds_new = model_0(x_test)

plot_predictions(predictions=y_preds)
plot_predictions(predictions=y_preds_new)
print(model_0.state_dict())

print(epoch_count, loss_values, test_loss_values)

#plot the loss curves
#dont forget to convert the values to be numpy useable before using matplotlib, which requires the tensor to be converted to numpy formate and the values be on the cpu instead of gpu
def plot_loss_curve():
    plt.figure() #this just creates a new figure instead of plotting on top of the old figure
    plt.plot(epoch_count, np.array(torch.tensor(loss_values).cpu().numpy()), label="Train loss")
    plt.plot(epoch_count, np.array(torch.tensor(test_loss_values).cpu().numpy()), label="Test loss")
    plt.title("training and test loss curves")
    plt.ylabel("Loss")
    plt.xlabel("epochs")
    plt.legend()
    plt.savefig("testLF.png")

plot_loss_curve()

#TODO: writing code to save a python model

#NOTE: there are three main methods for saving and loading in PyTorch.
#1. torch.save(), saves the model in python pickle format, pickle implemnets a binary protocl for serialize and de-serializing a python object structure, pickling is the process whereby a python object hierarchy is converted into a byte stream, and unpickling is the inverse operation, this is also known as serilizing, marshalling or flattening
#2. torch.load(), allows you to load a saved PyTorch object
#3. torch.nn.Module.load_state_dict() , this allows to load a model's saved state dictionary
# a state_dict is the learnable parameters, its a dictionary object that maps each layer to its parameter tensors, only learnable parameters, registered buffers and optimizers objects jave entries in teh state dict, with the latter containing information about the optimizer's state as well as the hyperparameters used

#NOTE: there are two ways to save, either save the entire model or just the state dict
 # - if only saving the state_dict, you would need to recreate the model instance- this is also the smaller file option
 # - if saving the entire model, you would need to exactly recreate the entire file structure including the source code of the model on the target syste,.

#saving the PyTorch Model

from pathlib import Path

#1. create a model directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

#2. create a model save path
MODEL_NAME = "01_pytorch_workflow_model_0.pth" #pytorch objects usually have the extension as .pt or .pth 
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(MODEL_SAVE_PATH)

#3. save the model_state_dict()

print(f"saving model to : {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)
#torch.save(obj (saved object), 
#          f (a file like object, has to implement write and flush, or a sting or os.PathLike object cintaining file name),
#          pickle_module=pickle,
#          pickle_protocol=DEFAULT_PROTOCOL, (module used for pickling metadata and objects)
#          _use_new_zipfile_serialization=True) can be specified to override the default protocol
#


#TODO: loading a model from a file

#torch.load (f, map_location=None, pickle_module=pickle, **pickle_load_args) uses python's unpickling facilities but treates storages, which underlie tensors, sepcially. they are first
#deserialized on the cpu and are then moved to the devices they were saved from. if no such device available it will raise exception, optionof dynamically remappeing to a alternative set of devices using the map_location if devices not availablle

# since we saved the state_dict() rather than the entire model, we will create a new instance of our model class and load the saved state_dict() into that.

#to load in a saved state_dict() we have to instantiate a new instance of out model class

loaded_model_0 = LinearRegressionModel()

#load the saved state_dict() of model_0, this will update the new instance with the updated parameters

loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
print(loaded_model_0.state_dict())

#make some predictions with out loaded model

loaded_model_0.eval()

with torch.inference_mode():
    loaded_model_preds = loaded_model_0(x_test)

print(loaded_model_preds)

model_0.eval()

with torch.inference_mode():
    y_preds = model_0(x_test)

print(y_preds == loaded_model_preds)
