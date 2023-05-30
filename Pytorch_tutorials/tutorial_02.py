import torch
import numpy as np

x = torch.empty(2, 3) # Creates a tensor of size 2 x 3
# print(x)

x = torch.rand(2,2)
# print(x)

x = torch.zeros(2,2)
# print(x)

x = torch.ones(2,2)
# print(x)
# print(x.dtype) # By default shows float32

"We can also explicitly specify the type"
x = torch.ones(2,2, dtype=torch.int)
# print(x.dtype)

"Creating tensor from python list"
x = torch.tensor([2.5,7])
# print(x)

"Performing basic arithmetic operation on tensors"
# operator + is used to perform element wise addition between two tensors.
# same operation can also be performed using torch.add(x,y) function for two tensors x and y.
# y.add_(x) performs the inplace addition of x and y and stores the result in y.
# In particular, whenever a fucntion contains trailing underscore(_), it performs inplace operation.

# Similarly, element wise multiplication, subtraction and division can also be performed.

"Slicing operations on tensor"
x = torch.rand(5,3)
# print(x)
# print(x[:,0]) # Prints the first column
# print(x[1,:]) # Print the second row
# print(x[1,1].item()) # To get the numeric value of single value present in the tensor.

"Reshaping the tensors"
x = torch.rand(4,4)
# print(x)
y = x.view(16) # Convert into one dimensional tensor
# print(y)
# If we do not want the single dimensional we can pass first parameter as -1
y = x.view(-1,8)
# print(y)

"Converting tensors to numpy array"
# a = torch.ones(5)
# print(a)
# b = a.numpy()
# print(b)
# They points to same memory location on CPU and hence modifying one changes other too.
# a+=1
# print(a)
# print(b)

"Converting numpy arrays to tensors"
# a = np.ones(5)
# b = torch.from_numpy(a)
# print(b)
# They points to same memory location on CPU and hence modifying one changes other too.
# a+=1
# print(a)
# print(b)

