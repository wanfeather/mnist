import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
#import torchvision.utils as vutils
import torchvision

#import matplotlib.pyplot as plt

#from model import Net

def save_model(model, num):
    print('Saving model...')
    torch.save(model.module.state_dict(), 'model/model_{}.pth'.format(num))
    print('Model {} saved\n'.format(num))

def check_accuracy(loader, model, compute_deivce):
    num_correct = 0
    num_samples = 0
    
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(compute_deivce)
            y = y.to(compute_deivce)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        accuracy = float(num_correct)/float(num_samples)*100.0
        
        print('Got {} / {} with accuracy {}%'.format(num_correct, num_samples, accuracy))

    model.train()

    return accuracy

def train():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081, ))
    ])

    train_dataset = datasets.MNIST(root = './data', train = True, transform = transform)
    test_dataset = datasets.MNIST(root = './data', train = False, transform = transform)
    train_dataloader = DataLoader(train_dataset, 4096, True)
    test_dataloader = DataLoader(test_dataset, 4096, False)

    num_epoches = 100
    lr = 1e-3
    device = torch.device('cuda')

    model = torchvision.models.resnet18(pretrained = False)
    model.conv1 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = (7, 7), stride = (2, 2), padding = (3, 3), bias = False)
    model.fc = nn.Linear(512, 10)
    model = model.to(device)
    model = nn.DataParallel(model, device_ids = [0, 1])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)

    model.train()
    with open('loss.csv', 'w') as f:
        for epoch in range(num_epoches):
            losses = []

            for batch_idx, (data, target) in enumerate(train_dataloader):
                data = data.to(device)
                target = target.to(device)

                optimizer.zero_grad()
                target_pre = model(data)
                loss = criterion(target_pre, target)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()

            avg_loss = sum(losses)/len(losses)
            print('\nCost at epoch {} is : {}'.format(epoch, avg_loss))
            f.write('{},{},'.format(epoch, avg_loss))
        
            print('Checking accuracy on Training Set')
            train_acc = check_accuracy(train_dataloader, model, device)
            f.write('{},'.format(train_acc))

            print('Checking accuracy on Testing set')
            test_acc = check_accuracy(test_dataloader, model, device)
            f.write('{}\n'.format(test_acc))

            if (epoch+1) % 10 == 0:
                save_model(model, epoch)

if __name__ == '__main__':
    train()
