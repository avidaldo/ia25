import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    """
    A lightweight ResNet-9 architecture suitable for CIFAR-10.
    Achieves high accuracy (>85%) quickly.
    """
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        
        # Initial block: 3 -> 64 (CIFAR-10 images are 32x32)
        self.prep = conv_block(in_channels, 64)
        
        # Layer 1: 64 -> 128 + pooling
        self.layer1 = conv_block(64, 128, pool=True) # Output: 128 x 16 x 16
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        # Layer 2: 128 -> 256 + pooling
        self.layer2 = conv_block(128, 256, pool=True) # Output: 256 x 8 x 8
        
        # Layer 3: 256 -> 512 + pooling
        self.layer3 = conv_block(256, 512, pool=True) # Output: 512 x 4 x 4
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        # Classifier
        self.classifier = nn.Sequential(nn.MaxPool2d(4), # Output: 512 x 1 x 1
                                        nn.Flatten(), 
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))
        
    def forward(self, xb):
        out = self.prep(xb)
        
        out = self.layer1(out)
        out = self.res1(out) + out
        
        out = self.layer2(out)
        
        out = self.layer3(out)
        out = self.res2(out) + out
        
        out = self.classifier(out)
        return out
