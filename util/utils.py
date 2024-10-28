from collections import OrderedDict
import torch
from util.load_data import load_data

from torchvision.transforms import Compose, Normalize, ToTensor
from util.target_network import Net

from flwr.common import parameters_to_ndarrays
from random import choices

DEVICE = torch.device("cuda:"+str(choices(population=[0,1], weights=[0.5, 0.5], k=1)[0]))

def train(network, train_loader, epochs, device: torch.device = torch.device("cuda")):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
    # optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
    for _ in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(network(images), labels)
            loss.backward()
            optimizer.step()

def test(net, test_loader, steps: int = None, device: torch.device = torch.device("cuda")):
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss()
    total, correct, loss = 0, 0, 0.0
    net.eval()

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)

            loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += images.size(0)
            if steps is not None and batch_idx == steps:
                break

    return loss, correct / float(total)

def test_on_server(parameters):
    """
    Test code for the server. Test code loading functionality should be here.
    Should modify the test data load part
    If you do not want to use this functionality, just return 0, {"accuracy": 0}
    """
    # Test data load
    param_array = parameters_to_ndarrays(parameters)
    _, test_loader, _ = load_data()

    net = Net()
    params_dict = zip(net.state_dict().keys(), param_array)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

    loss, accuracy = test(net, test_loader)

    return loss, {"accuracy" : accuracy}
