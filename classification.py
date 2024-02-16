
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import time

device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"learning will be run on : {device}")

#torch.set_default_device("cuda")

raw_data_train = datasets.FashionMNIST(
    root="raw_data", 
    download=True, 
    train=True, 
    transform=ToTensor()
)

raw_data_test = datasets.FashionMNIST(
    root="raw_data", 
    download=True, 
    train=False, 
    transform=ToTensor()
)

batch_size = 32

data_train = DataLoader(
    dataset=raw_data_train, 
    batch_size=batch_size
)

data_test = DataLoader(
    dataset=raw_data_test, 
    batch_size=batch_size
)

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(28*28, 512), 
            nn.ReLU(),
            nn.Linear(512, 512), 
            nn.ReLU(),
            nn.Linear(512, 10), 
        )

    def forward(self, X):
        X = self.flatten(X)
        return self.linear_stack(X)

model = Model().to(device)

loss_fn = nn.CrossEntropyLoss()
optimization = torch.optim.SGD(
    model.parameters()
)

def train(model : Model, dataset, loss_fn : nn.CrossEntropyLoss, optimization : torch.optim.SGD):
    model.train(True)
    for X, y in dataset:
        X = X.to(device)
        y = y.to(device)

        prediction = model(X)
        loss = loss_fn(prediction, y)
        #backwards
        loss.backward()               
        optimization.step()
        optimization.zero_grad() # clear

def test(model, dataset, loss_fn : nn.CrossEntropyLoss):
    model.train(False)
    accuraccy = 0.0        
    for X, y in dataset:
        X = X.to(device)
        y = y.to(device)

        prediction = model(X)
        prediction = prediction.argmax(1)
        equal = y == prediction
        equal = equal.type(torch.float)
        accuraccy += equal.sum().item()
    accuraccy /= len(dataset.dataset)
    print(f"accuraccy : {round(100 * accuraccy, 2)}%")

time_start = time.time()
batchs = 10
for batch in range(batchs):
    print(f"batch : {batch}")
    train(model, data_train, loss_fn, optimization)
    test(model, data_train, loss_fn)
time_stop = time.time()
print(f"total time : {round(time_stop-time_start, 2)}s")




