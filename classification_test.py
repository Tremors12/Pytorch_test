import torch
from torch import nn 
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"device : {device}")

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
print("Loading model . . .")
model.load_state_dict(torch.load("model.pth"))

loss_fn = nn.CrossEntropyLoss()
optimization = torch.optim.SGD(
    model.parameters()
)


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


print("Testing model accuracy")
test(model, data_train, loss_fn)

