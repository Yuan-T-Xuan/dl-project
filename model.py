import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    if type(m) in [nn.Conv2d, nn.ConvTranspose2d, nn.Linear]:
        nn.init.normal(m.weight, mean=0, std=0.02)

def residual(conv1, conv2, norm1, norm2, input_tensor):
    x = F.relu(norm1(conv1(input_tensor)))
    x = norm2(conv2(x))
    return F.relu(x + input_tensor)

class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.c1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3)
        self.norm_c1 = nn.InstanceNorm2d(64, affine=True)
        self.d1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.norm_d1 = nn.InstanceNorm2d(128, affine=True)
        self.d2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.norm_d2 = nn.InstanceNorm2d(256, affine=True)
        #
        self.r1_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.norm_r1_1 = nn.InstanceNorm2d(256, affine=True)
        self.r1_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.norm_r1_2 = nn.InstanceNorm2d(256, affine=True)
        #
        self.r2_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.norm_r2_1 = nn.InstanceNorm2d(256, affine=True)
        self.r2_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.norm_r2_2 = nn.InstanceNorm2d(256, affine=True)
        #
        self.r3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.norm_r3_1 = nn.InstanceNorm2d(256, affine=True)
        self.r3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.norm_r3_2 = nn.InstanceNorm2d(256, affine=True)
        #
        self.r4_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.norm_r4_1 = nn.InstanceNorm2d(256, affine=True)
        self.r4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.norm_r4_2 = nn.InstanceNorm2d(256, affine=True)
        #
        self.r5_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.norm_r5_1 = nn.InstanceNorm2d(256, affine=True)
        self.r5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.norm_r5_2 = nn.InstanceNorm2d(256, affine=True)
        #
        self.r6_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.norm_r6_1 = nn.InstanceNorm2d(256, affine=True)
        self.r6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.norm_r6_2 = nn.InstanceNorm2d(256, affine=True)
        #
        self.u1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        self.norm_u1 = nn.InstanceNorm2d(128, affine=True)
        self.u2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, output_padding=1)
        self.norm_u2 = nn.InstanceNorm2d(64, affine=True)
        self.c2 = nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=1)
        self.norm_c2 = nn.InstanceNorm2d(3, affine=True)
    
    def forward(self, x):
        x = F.relu(self.norm_c1(self.c1(x)))
        x = F.relu(self.norm_d1(self.d1(x)))
        x = F.relu(self.norm_d2(self.d2(x)))
        x = residual(self.r1_1, self.r1_2, self.norm_r1_1, self.norm_r1_2, x)
        x = residual(self.r2_1, self.r2_2, self.norm_r2_1, self.norm_r2_2, x)
        x = residual(self.r3_1, self.r3_2, self.norm_r3_1, self.norm_r3_2, x)
        x = residual(self.r4_1, self.r4_2, self.norm_r4_1, self.norm_r4_2, x)
        x = residual(self.r5_1, self.r5_2, self.norm_r5_1, self.norm_r5_2, x)
        x = residual(self.r6_1, self.r6_2, self.norm_r6_1, self.norm_r6_2, x)
        x = F.relu(self.norm_u1(self.u1(x)))
        x = F.relu(self.norm_u2(self.u2(x)))
        x = torch.tanh(self.norm_c2(self.c2(x)))
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
