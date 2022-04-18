import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.models import resnet18

import matplotlib.pyplot as plt

from model import Net

from train_mnist import check_accuracy

class Mask():
    def __call__(self, x):
        index = torch.randint(0, 15, (2, ))
        x[:, index[0]:index[0]+14, index[1]:index[1]+14] = 0
        #x[:, 0:14, 0:14] = 0
        
        return x

def visualize():
    tf= transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081, )),
        Mask()
    ])

    train_dataset = datasets.MNIST(root = './data', train = True, transform = tf)
    test_dataset = datasets.MNIST(root = './data', train = False, transform = tf)
    train_loader = DataLoader(dataset = train_dataset, batch_size = 4096, shuffle = False, num_workers = 2)
    test_loader = DataLoader(dataset = test_dataset, batch_size = 4096, shuffle = False, num_workers = 2)


    print('Loading model...')
    device = torch.device('cuda')
    #model = Net(1, 10).to(device)
    model = resnet18(pretrained = False)
    model.conv1 = torch.nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = (7, 7), stride = (2, 2), padding = (3, 3), bias = False)
    model.fc = torch.nn.Linear(512, 10)
    model = model.to(device)
    model.load_state_dict(torch.load('ResNet_mnist.pth', map_location = device))
    print('Model loaded.\n')

    model.eval()


    check_accuracy(train_loader, model, device)
    check_accuracy(test_loader, model, device)
    
    #with torch.no_grad():
        #for x, y in test_loader:
            #x, y = x.to(device), y.to(device)
            #scores = model(x)
            #_, pred = scores.max(1)
            #print(pred)

            #grid = vutils.make_grid(x, nrow = 10, padding = 0)
            #grid = grid.permute(1, 2, 0).cpu()
            #plt.imshow(grid)
            #plt.show()
            #plt.savefig('result.png')
    

if __name__ == '__main__':
    visualize()
