import torch
from torch import nn
from utils import initialize_weights


class ARCNN(nn.Module):
    def __init__(self):
        super(ARCNN, self).__init__()
        self.base = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU(),
            nn.Conv2d(64, 32, kernel_size=7, padding=3),
            nn.PReLU(),
            nn.Conv2d(32, 16, kernel_size=1),
            nn.PReLU()
        )
        self.last = nn.Conv2d(16, 3, kernel_size=5, padding=2)

        self._initialize_weights()

    def forward(self, x):
        x = self.base(x)
        x = self.last(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            initialize_weights(m)


class FastARCNN(nn.Module):
    def __init__(self):
        super(FastARCNN, self).__init__()
        self.base = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=2, padding=4),
            nn.PReLU(),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=7, padding=3),
            nn.PReLU(),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.PReLU()
        )
        self.last = nn.ConvTranspose2d(64, 3, kernel_size=9, stride=2, padding=4, output_padding=1)

        self._initialize_weights()

    def forward(self, x):
        x = self.base(x)
        x = self.last(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            initialize_weights(m)

class DnCNN(nn.Module):
    def __init__(self, depth=15, n_channels=64, image_channels=3, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(
            nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            layers.append(
                nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                          bias=True))
            #layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding,
                      bias=True))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y + out

    def _initialize_weights(self):
        for m in self.modules():
            initialize_weights(m)

# convolutional block contains 4 convolutional layers and three relu activation function after first 3 conv layers           
class Conv_Block(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(Conv_Block, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1,bias=True))
        layers.append(nn.ReLU(inplace=True))
        for i in range(2):
            layers.append(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1,bias=True))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1,bias=True))
        self.conv_block = nn.Sequential(*layers)
        
    def forward(self, x):
        conv_out = self.conv_block(x)
        return conv_out

# Densely connected residual network which ultilizes the Conv_Block and make dense connection between them
class DenseResNet(nn.Module): 
    def __init__(self): 
        super(DenseResNet, self).__init__() 

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, padding = 1, bias = True) 
        self.relu1 = nn.ReLU(inplace=True)
        
        # Make Conv Blocks 
        self.convblock1 = self._make_Conv_block(Conv_Block, 32,64) 
        self.convblock2 = self._make_Conv_block(Conv_Block, 96,64)
        self.convblock3 = self._make_Conv_block(Conv_Block, 160,64)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1,bias=True)
        self._initialize_weights()
        
    def _make_Conv_block(self, block, in_channels,out_channels): 
        layers = [] 
        layers.append(block(in_channels,out_channels)) 
        return nn.Sequential(*layers) 
    
    def forward(self, x): 
        y = x
        conv_out = self.relu1(self.conv1(x))
        conv1_out = self.convblock1(conv_out)
        dense1 = self.relu1(torch.cat([conv_out,conv1_out],1))
        conv2_out = self.convblock2(dense1)
        dense2 = self.relu1(torch.cat([conv_out,conv1_out,conv2_out],1))
        conv3_out = self.convblock3(dense2)
        out = self.conv2(self.relu1(conv3_out))
        return y + out
        
    def _initialize_weights(self):
        for m in self.modules():
            initialize_weights(m)
