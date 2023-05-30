import torch
import torch.nn as nn
import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print("Softmax numpy: ", outputs)

x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x,dim=0)
print("Softmax tensor: ", outputs)

def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss / float(predicted.shape[0])

# y must be one hot encoded
# If class 0: [1 0 0]
# If class 1: [0 1 0]
# If class 2: [0 0 1]
Y = np.array([1, 0, 0])

# Y_Predicted and Probabilities
Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.1, 0.3, 0.6])
l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)
print(f'Loss1 numpy: {l1:.4f}')
print(f'Loss2 numpy: {l2:.4f}')

'''
To apply the cross entropy loss function in pytorch,
Be careful about the following points

nn.CrossEntropyLoss applies nn.LogSoftmax + nn.NLLLoss(Negative Log Likelihood Loss)
--> No Softmax in last layer!

Y has class labels, not One-Hot encoded!
Y_pred has raw scores (logits), no Softmax!

'''

loss = nn.CrossEntropyLoss()

Y = torch.tensor([0])
# size = n_samples x n_classes = 1 x 3
Y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])
Y_pred_bad = torch.tensor([[0.5, 2.0, 0.3]])

l1 = loss(Y_pred_good,Y)
l2 = loss(Y_pred_bad,Y)

print(l1.item())
print(l2.item())

_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)
print(predictions1)
print(predictions2)

"For multiple samples , say 3 in our example"

Y = torch.tensor([2,0,1])
# size = n_samples x n_classes = 3 x 3
Y_pred_good = torch.tensor([[0.1, 1.0, 2.1], [2.0, 1.0, 0.1], [0.1, 3.0, 0.1]])
Y_pred_bad = torch.tensor([[2.1, 1.0, 0.1], [0.1, 1.0, 2.1], [0.1, 3.0, 0.1]])

l1 = loss(Y_pred_good,Y)
l2 = loss(Y_pred_bad,Y)

print(l1.item())
print(l2.item())

_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)
print(predictions1)
print(predictions2)



