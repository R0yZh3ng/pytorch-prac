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



