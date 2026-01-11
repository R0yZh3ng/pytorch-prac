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
optimzer = torch.optim.SGD(params=model_0.parameters(), lr=0.01) #stochastic gradient descent, perform gradient descent by randomly adjusting values and following a direction that lowers the loss function
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
epochs = 1

#0. loop through the data
for epoch in range(epochs):
    model_0.train() #train mode in PyTorch set all parameters taht require gradients to require gradients

    model_0.eval() #turns off graident tracking


 

