# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimizer 
# 3) Training Loop
#    - forward pass: Compute prediction 
#    - backward pass: Gradients
#    - update weights

import torch
import torch.nn as nn

# f = 2 * x
X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([2,4,6,8], dtype=torch.float32)

w = torch.tensor(0.1,dtype=torch.float32, requires_grad=True)

# Model Prediction
def forward(x):
    return w * x

print(f'Prediction before training: f(5) = {forward(5):.4f}')

# Training
learning_rate = 0.01
n_iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD([w], lr=learning_rate)

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients = backward pass
    l.backward() # dl/dw

    # update weights
    optimizer.step()

    # Zero Gradients
    optimizer.zero_grad()

    if epoch % 10 ==0:
        print(f'epoch {epoch+1}: w = {w:4f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.4f}')