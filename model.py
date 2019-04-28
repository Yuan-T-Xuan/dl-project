import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    if type(m) in [nn.Conv2d, nn.ConvTranspose2d, nn.Linear]:
        nn.init.normal(m.weight, mean=0, std=0.02)


class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = nn.Conv2d(32, 128, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2d(128, affine=True)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.norm3 = nn.InstanceNorm2d(256, affine=True)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.norm4 = nn.InstanceNorm2d(128, affine=True)
        self.deconv2 = nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.norm5 = nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        x = self.norm1(F.relu(self.conv1(x)))
        x = self.norm2(F.relu(self.conv2(x)))
        x = self.norm3(F.relu(self.conv3(x)))
        x = self.norm4(F.relu(self.deconv1(x)))
        x = self.norm5(F.relu(self.deconv2(x)))
        x = torch.tanh(self.deconv3(x))
        return x

class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = nn.Conv2d(32, 128, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2d(128, affine=True)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.norm3 = nn.InstanceNorm2d(256, affine=True)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=24, stride=1)
        self.norm4 = nn.InstanceNorm2d(512, affine=True)
        self.out = nn.Linear(512, 1)
    
    def forward(self, x):
        x = self.norm1(F.leaky_relu(self.conv1(x)))
        x = self.norm2(F.leaky_relu(self.conv2(x)))
        x = self.norm3(F.leaky_relu(self.conv3(x)))
        x = self.norm4(F.leaky_relu(self.conv4(x))).view(-1, 512)
        return torch.sigmoid(self.out(x))
