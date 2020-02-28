import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import torch.optim as optim
import argparse
import sys
import os
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import io, transform

from my_net import Net


def parse():
    parser = argparse.ArgumentParser(description='Simple Neural Network Model')
    subparsers = parser.add_subparsers(help='select neural network mode')
    parser_train = subparsers.add_parser('train', help='training mode')
    parser_test = subparsers.add_parser('test', help='testing mode')
    #parser.add_argument('mode', type=str,
    #                help='select whether neural network is in test/train mode')
    parser_test.add_argument('image', type=str, help='specify the test image')
    args = parser.parse_args()
    var = vars(args)
    if 'image' not in var:
        mode = 'train'
        img = ''
    else:
        mode = 'test'
        img = var['image']
    if mode == 'train' and 'image' in var:
        print("Warning: Images submitted under train mode are ignored")
    if mode == 'test' and args.image != "":
        print("Testing mode activated with image: {0:}".format(args.image))
    else:
        print("Training mode activated")
    return mode, img

def load_data():
    train_data = torchvision.datasets.CIFAR10(
    root='./data.cifar10',                          # location of the dataset
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to torch.FloatTensor of shape (C x H x W)
    download=True                                   # if you haven't had the dataset, this will automatically download it for you
    )

    train_data_loader = Data.DataLoader(dataset=train_data, batch_size=128, shuffle=True, num_workers=2)
    test_data = torchvision.datasets.CIFAR10(root='./data.cifar10/', train=False, transform=torchvision.transforms.ToTensor())
    test_data_loader = Data.DataLoader(dataset=test_data, batch_size=128, shuffle=False, num_workers=2)
    return train_data_loader, test_data_loader


def train(trainloader, testloader, net, criterion, optimizer, device):
    print("{0:>4s}   {1:>14s} {2:>12s} {3:>11s} {4:>10s}".format("Loop", "Train Loss", "Train Acc %", "Test Loss", "Test Acc %"))
    PATH = './model/cifar_net.pth'
    length = 500
    net.train()
    for epoch in range(length):  # loop over the dataset multiple times
        running_loss = 0.0
        training_loss = 0.0
        min_loss = math.inf
        total_train = 0
        correct_train = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
    
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            training_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                #print('[%d, %5d] loss: %.3f' %
                #    (epoch + 1, i + 1, running_loss / 2000))
                if min_loss > running_loss:
                    torch.save(net.state_dict(), PATH)
                    min_loss = running_loss
                running_loss = 0.0

        testing_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                testing_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
        torch.save(net.state_dict(), PATH)
        print("{0:>2d}/{1:d} {2:>15f} {3:>12f} {4:>11f} {5:>10f}".format(epoch, length, training_loss / len(trainloader), 100 * correct_train / total_train, testing_loss / len(testloader), 100 * correct / total)) 


def test(image_path, net, device, classes):
    net.eval()
    net.to(device)
    img = Image.open(image_path)
    img = TF.to_tensor(img)
    if img.shape[1] != 32 or img.shape[2] != 32:
        img = transform.resize(img, (1, 3, 32, 32), mode='constant', anti_aliasing=True, anti_aliasing_sigma=None)
        img = torch.from_numpy(img)
        print("Warning: Incorrect image input dimensions. Resizing using scalar interploation.")
    else:
        img = transform.resize(img, (1, 3, 32, 32), mode='constant', anti_aliasing=True, anti_aliasing_sigma=None)
        img = torch.from_numpy(img)
    img = img.to(device)
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    net.conv1.register_forward_hook(get_activation('conv1'))
    output = net(img)
    _, predicted = torch.max(output.data, 1)
    print("prediction result: ", classes[predicted])
    act = activation['conv1'].squeeze()
    act = act.cpu()
    fig, axes = plt.subplots(6, 6, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.025, wspace=0.025, top=0.95, bottom=0.05, left=0.05, right=0.95)
    for idx in range(36):
        row_ = idx // 6
        col_ = idx % 6
        ax = axes[row_, col_]
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        if idx < act.size(0):
            ax.imshow(act[idx], cmap='gray')
    plt.savefig('conv1.png')


if __name__ == '__main__':
    if not os.path.exists('model'):
        try:
            os.makedirs('model')
        except OSError as e:
            raise Exception("Cannot create directory")
    mode, image_path = parse()
    PATH = './model/cifar_net.pth'
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net()
    net.to(device)
    if mode == 'train':
        train_loader, test_loader = load_data()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.5)
        train(train_loader, test_loader, net, criterion, optimizer, device)
    else:
        net.load_state_dict(torch.load(PATH))
        test(image_path, net, device, classes)
