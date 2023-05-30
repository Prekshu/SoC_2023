import torch

# f = 2 * x
X = torch.tensor([1,2,3,4])
Y = torch.tensor([2,4,6,8])

w = torch.tensor(0.1, requires_grad=True)

# Model Prediction
def forward(x):
    return w * x

# loss = MSE
def loss(y,y_predicted):
    return ((y_predicted-y)**2).mean()


print(f'Prediction before training: f(5) = {forward(5):.4f}')

# Training
learning_rate = 0.01
n_iters = 100

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients = backward pass
    l.backward() # dl/dw

    # update weights
    with torch.no_grad():
        w -= learning_rate * w.grad

    # Zero Gradients
    w.grad.zero_()
    if epoch % 10 ==0:
        print(f'epoch {epoch+1}: w = {w:4f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.4f}')