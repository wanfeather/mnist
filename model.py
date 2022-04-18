import torch.nn as nn

class Net(nn.Module):
    
    def __init__(self, input, output):
        super(Net, self).__init__()

        self.conv = nn.Sequential(
            Conv(input, 100, 3, (1, 3)),
            Conv(100, 225, 1, (3, 1)),
            Conv(225, 400, 1, (1, 3)),
            Conv(400, 1225, 1, (3, 1))
        )
        self.fc = nn.Linear(4900, output)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class Conv(nn.Module):

    def __init__(self, input, output, conv_kernel_size, pool_kernel_size):
        super(Conv, self).__init__()

        self.conv = nn.Conv2d(in_channels = input, out_channels = output, kernel_size = conv_kernel_size, bias = False)
        self.pool = nn.AvgPool2d(pool_kernel_size)
        self.act_f = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.act_f(x)
        x = self.pool(x)

        return x
