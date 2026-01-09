import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


##introduction to Tensors##

#creating tensors#

#scalar#

scalar = torch.tensor(7)

#pytorch tensors are created using torch.tensor

print(scalar.ndim) # ndim gives the dimension
#0

#gets tensor back as a python int
print(scalar.item)

#vector
vector = torch.tensor([7, 7]) 
print(vector)
#tensor([7, 7])

print(vector.ndim)
#1

print(vector.shape)
#torch.Size([2])

#matrix
MATRIX = torch.tensor([[7, 8], [9, 10]])
print(MATRIX)
#tensor([7, 8],[9, 10])

print(MATRIX.ndim)
#2

print(MATRIX[1])
#tensor([9, 10])

print(MATRIX.shape)
#torch.Size([2, 2])

#TENSOR

TENSOR = torch.tensor([[[1,2,3],[3, 6, 9], [2, 4,5 ]]]) 
print(TENSOR)
#tensor([[[1, 2, 3],[3, 6, 9],[2, 4, 5]]])

print(TENSOR.ndim)
#3

print(TENSOR.shape)
#torch.Size([1, 3, 3]) # one, three by three shaped tensor
#count it in the dimensions of how many brackets deep the tensor is in terms of index of the shape
#tensor can be able shape and size, determined by the problem



#NOTE - 
#scalars and vectors are lowercase, matrix and Tensors are Uppercase

### Random Tensors

#why random tensors?

#random tensors are important because the way many neural networks learn is that they start with tensors full of random numbers and then adjust those random numbers to better represent the data
#start with random numbers -> look at data -> update random numbers -> look at data -> update random numbers (so this is basically gradient descent)

#create a random tensor of size (3, 4)
random_tensor = torch.rand(3, 4)
#torch.rand(*size, *, out=None, dtype= None,layout=torch.strided, device=None), requires_grad_False)->Tensor


print(random_tensor)


#create a random tensor with similar shape to an iumage tensor
random_image_tensor = torch.rand(size=(224, 224, 3)) #height, width, color channel
print(random_image_tensor.shape)
print(random_image_tensor.ndim)


#zeros and ones
#create a tensor of all zeros
zeros = torch.zeros(size=(3, 4))
print(zeros)

print(zeros*random_tensor) # this also works to convert a tensor into zeros, this is just matrix operations where every element is multiplied by the zeros to create all the zeros


#create a tensor of all ones
ones = torch.ones(size=(3, 4))

print(ones)

print(ones.dtype) #this just returns the data type, it is automaticlaly initialized as torch.float32 if not specified otherwise

#creatinga range of tensors and tensors-like
#use torch.range()
zero_to_nine = torch.arange(0, 10)
print(zero_to_nine)
#torch.arange(start=0, end, step = 1, *, out=None, dtype=None, layout=torch.strided, device =None, requires_grad= False) ->tensor
#finishes at end-1, so need ot account for that if includingt the entire range its basically [begin, end)
nine_zeros = torch.zeros_like(input = zero_to_nine)
print(nine_zeros) #creates a tensor like the specified input tensor

#Tensor datatypes

#float32 TENSOR

float_32_tensor = torch.tensor([3.0, 6.0, 9,0], 
                               dtype = None, #data type, can use half tensor for faster calculation with less precision, or double tensor for more precision slower calculation. the default floatTensor is 32bit 
                               device =None, #what device your tensor is on 
                               requires_grad = False) #whether or not to track graidents with this tensor's operations 
#these fields are the most importatn fields of a tensor
print(float_32_tensor)

#Note: tensor datatypes is one of the 3 big errors youll run into with pytorch and deep learning
        #1.tensor is not rigth data type
        #2.tensor not right shape
        #3.tensor not on the right device - living on cpu compared to gpu -etc

float_16_tensor = float_32_tensor.type(torch.half) # or torch.float16 #changes tensor dtype
print(float_16_tensor)

### getting infromation from tensors
#tensor.dtype
#tensor.shape
#tensor.device

print(f"Datatype of Tensor: {float_32_tensor.dtype}")
print(f"shape of tensor: {float_32_tensor.shape}")
print(f"device the tensor is on: {float_32_tensor.device}")

#### Manipulating tensors (tensor operations)

#addition
#subtraction
#mutiplication
#division
#matrix multiplication


#create a tensor
tensor = torch.tensor([1,2,3])
tensor_add = tensor + 10

print(tensor_add)

#multiply tensor by 10
tensor_mult = tensor * 10 #this is a  scalar multiplcation- element wise
print(tensor_mult)

tensor_sub = tensor - 10
print(tensor_sub)

torch.mul(tensor, 10)
torch.add(tensor, 10)# these also work and are provided by the torch library

## matrix multiplication (dot product)
#tensor * tensor would result in the elemnt by element product of the tensor, so each element is applied to the corresponding element
#i.e. [1, 2, 3] * [1, 2, 3] = [1, 4, 9]

tensor_ew = tensor*tensor
print(tensor_ew)

dot_product = torch.matmul(tensor, tensor) # pytorch provided matrix multiplciation and returns the dot product of the tensor
print(dot_product)
dot_product = tensor @ tensor # also works, just less clear when code get large
#dot_product = torch.mm(tensor, tensor) # this the same thing, short version of torch.matmul() but is a lot stricter and enforce inner dimension matches, can use this if know for sure multiplying matrixes and when writing lower level code

#can write the dot product function yourself using
value = 0
for i in range(len(tensor)):
    value += tensor[i] * tensor[i]
print(value)

#this will result in the same answer but the pytorch provided function is a lot faster, (almost 10x), so given any operations that are provided by the pytorch library, use it, it will reliably be faster than you own impelmentation unless you are writting a library in a lower level language

###one of the most common errors in deep learning: shape errors ***

##there are two rules that performing matrix multiplcation needs to satisfy

# 1, the inner dimensions must match
# ex : (3, 2) @ (3, 2) will not work
#      (3, 2) @ (2, 3) will work
#      this is just the basics of matrix algebra which is that for matrix multiplication to work, the number of columns of the first matrix must match the number of rows of the second matrix
# 2. the resulting matrix will have the shape of the outter dimensions
#
#       (3, 2) @ (2, 3) shape = (3, 3)
#       (3, 2) @ (2, 4) shape = (3. 4)
#       \
tensor = torch.tensor([[1,2,3],
                       [4,5,6]])

#### transpose ######
# if in the case which the matrix shapes do not allow for inner dimensions ot match, you can use transpose to switch the dimensions of a tensor from n x m to m x n , whihc may or may not resolve the shape mismatch
print(tensor)
tensor.T # need to use these within a funcion or assign them to a value for them to work
print(tensor)
#The use of `x.T` on tensors of dimension other than 2 to reverse their shape is deprecated and it will throw an error in a future release. Consider `x.mT` to transpose batches of matrices or `x.permute(*torch.arange(x.ndim - 1, -1, -1))` to reverse the dimensions of a tensor. (Triggered internally at /pytorch/aten/src/ATen/native/TensorShape.cpp:4416.)
# tensor.T
tensor.mT
# tensor.permute(tensor.ndim -1, -1) 
print(tensor.mT) 

### finding the min , max, mean , sum etc (tensor aggregation)

#create a tensor 
x = torch.arange(0, 100, 10)
print(x)
max = torch.max(x)
print(max)
max_2 = x.max()
print(max_2)

mean = torch.mean(x.type(torch.float32)) # mean does not take long
print(mean)

mean_2 = x.type(torch.float32).mean()
print(mean_2)

torch.sum()
x.sum()

## positional min and max

##find the position in tensor that has the minimum/max value - bascially the index in which the min and max is at in the tensor
x.argmax()
x.argmin()

# tensor elements can be accessed through indexing just like arrays
x[9]



# reshaping, stacking, squeezing and unsqueezing tensors
#
# # reshaping - reshapes an input tensor to a defined shape
# # view - return a view of an input tnesor of 
