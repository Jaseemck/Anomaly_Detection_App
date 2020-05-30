# Anomaly Detection in IoT Devices

### Importance of Security in IoT devices
Cyberattacks in the Internet of Things(IoT) platform is a rising concern in the domain of IoT. IoT devices are increasingly used in smart homes, autonomous vehicles and smart appliances these days. Confidential data’s are involved in these applications. Hence the threats and attacks can have serious
consequences. Federated learning (FL) is a family of Machine Learning algorithms introduced by Google in 2016. In a network that consists of edge nodes connected to the central server, each of the nodes trains a local model and only that model is shared with the server, not the data. This is the core idea of federated learning. Hence privacy is preserved. The concept of federated learning can ultimately transform the ideology of privacy in the internet.

### Federated Learning
Federated learning is a machine learning technique where the connected devices collect data participates in training the central machine learning model. This technique is most popular with systems deployed at scale. Federated learning is used where the data should not be shared with the cloud but requires information and analysis from the data at hand. This method is used where confidential data are involved and is ideal for edge devices. Personalized keyboards on smartphones are one of the areas where it is first implemented. GBoard application is one example. The predictive features on those personalized keyboards learn from features such as your typing patterns, usage of words, slangs to give you better suggestions in the future. But the keyboard is used to type many private and confidential data. So, it is not secure to send those data directly to the cloud. Then it would be privacy invasion and a lot of our personal stuff can be leaked and misused.

[![What is Federated Learning?](https://raw.githubusercontent.com/Jaseemck/Anomaly_Detection_App/master/flpic1.jpeg)](https://www.youtube.com/watch?v=wOAkaxiCYnM)

The basic working of federated learning is as follows. Machine learning models are trained on your edge devices and then the stuff which can be either weights in neural networks or other types of machine learning models, are the only ones sent to the central server. Now the central server averages those stuff which it receives from connected edge devices and then uses them to train its central machine learning model. After it undergoes training up to some epochs, that central machine learning model is distributed back to the devices to be used for predictive purposes or for further training. In traditional machine learning approach, training a neural network would require to have a single copy of the model and all of the training data in one place. But in reality, data is mostly gathered across an array of sensors. In those scenarios, all sensor data would have to be sent to a central server fortraining and the resulting network weights distributed back to the sensors. However, these sensors often have limited bandwidth and intermittent connections to the central server.

![](https://raw.githubusercontent.com/Jaseemck/Anomaly_Detection_App/master/FLmodel.gif)


### Introduction to Pysyft
PySyft is a Python library for secure and private deep learning. PySyft requires Python >= 3.6 and PyTorch 1.1.0.

```
!pip install syft
import torch as torch
import syft as sy
hook = sy.TorchHook(torch)
```
This is done to override PyTorch’s methods to execute commands on one worker that are called on tensors controlled by the local worker. It also allows us to move tensors between workers.Virtual workers are entities present on our local machine. They are used to model the behavior of actual workers.

We can perform federated learning on client devices by following these steps:
* send the model to the device,
* do normal training using the data present on the device,
* get back the smarter model.

### Federated Learning using PySyft

Here, we implement the federated learning approach to train a simple neural network on the MNIST dataset using the two workers: Raj and kaif.

```python
import torch
import torchvision
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
```

In real-life applications, the data is present on client devices. To replicate the scenario, we send data to the **VirtualWorkers**.

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, )),
])

train_set = datasets.MNIST(
    "~/.pytorch/MNIST_data/", train=True, download=True, transform=transform)
test_set = datasets.MNIST(
    "~/.pytorch/MNIST_data/", train=False, download=True, transform=transform)

federated_train_loader = sy.FederatedDataLoader(
    train_set.federate((raj, kaif)), batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=64, shuffle=True)

```
The FederatedDataset class is intended to be used like the PyTorch’s Dataset class. Pass the created FederatedDataset to a federated data loader “FederatedDataLoader” to iterate over it in a federated manner. The batches then come from different devices.

```python
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = Model()
optimizer = optim.SGD(model.parameters(), lr=0.01)

```
Since the data is present on the client device, we obtain its location through the location attribute. The important additions to the code are the steps to get back the improved model and the value of the loss from the client devices.

```python
for epoch in range(0, 5):
    model.train()
    for batch_idx, (data, target) in enumerate(federated_train_loader):
        # send the model to the client device where the data is present
        model.send(data.location)
        # training the model
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # get back the improved model
        model.get()
        if batch_idx % 100 == 0:
            # get back the loss
            loss = loss.get()
            print('Epoch: {:2d} [{:5d}/{:5d} ({:3.0f}%)]\tLoss: {:.6f}'.format(
                epoch+1,
                batch_idx * 64,
                len(federated_train_loader) * 64,
                100. * batch_idx / len(federated_train_loader),
                loss.item()))
```


```
Epoch:  1 [    0/60032 (  0%)]	Loss: 2.306809
Epoch:  1 [ 6400/60032 ( 11%)]	Loss: 1.439327
Epoch:  1 [12800/60032 ( 21%)]	Loss: 0.857306
Epoch:  1 [19200/60032 ( 32%)]	Loss: 0.648741
Epoch:  1 [25600/60032 ( 43%)]	Loss: 0.467296
...
...
...
Epoch:  5 [32000/60032 ( 53%)]	Loss: 0.151630
Epoch:  5 [38400/60032 ( 64%)]	Loss: 0.135291
Epoch:  5 [44800/60032 ( 75%)]	Loss: 0.202033
Epoch:  5 [51200/60032 ( 85%)]	Loss: 0.303086
Epoch:  5 [57600/60032 ( 96%)]	Loss: 0.130088
```


```python
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        test_loss += F.nll_loss(
            output, target, reduction='sum').item()
        # get the index of the max log-probability
        pred = output.argmax(1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)

print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss,
    correct,
    len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
```

Test set: Average loss: 0.2428, Accuracy: 9300/10000 (93%)

That’s it. We have trained a model using the federated learning approach.

*More Tutorial are available *[here](https://github.com/OpenMined/PySyft/tree/master/examples/tutorials)


**Download our paper on Role of **[Federated Learning in IoT Anomaly Detection](https://github.com/Jaseemck/Anomaly_Detection_App/raw/master/FinalPaper.pdf)
