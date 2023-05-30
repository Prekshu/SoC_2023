import torch

"Whenever we want to find gradient of a tensor we need to pass the arguments requires_grad as True"
x = torch.randn(3, requires_grad=True)
# print(x)

y = x+2
# print(y)
z = y*y*2
z = z.mean()
# print(z)

"Uses computational graph and concept of back propagation for calculating derivatives"
# Since we have called z.mean() it is a scalar value and gradient can be directly calculated as below
z.backward() #dz/dx
# print(x.grad)

# However if we do not call mean() funtion and simply perform z.backward(), it will throw error.
z = y*y*2
# To calculate gradient of values which are not scalar, do the following:
v = torch.tensor([0.1,1.0,0.001], dtype = torch.float32)
z.backward(v)
# print(x.grad)
# The gradient is calculated by multplying the Jacobian matrix with the vector v.

"If we want to remove the gradient function attribute"
"To prevent gradient history in computational graph"
# First Option
x.requires_grad_(False)
# print(x)
# Second Option
y = x.detach()
# print(y)
# Third Option
with torch.no_grad():
    y = x + 2
    # print(y)

"Clearing weight gradients while training"
weights = torch.ones(4, requires_grad = True)
for epoch in range(3):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_() # This line is important

"Using optimizers"
optimizer = torch.optim.SGD(weights, lr=0.01)
optimizer.step()
optimizer.zero_grad()