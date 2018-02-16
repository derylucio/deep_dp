import torch.nn as nn
import os

cfg16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
os.environ['TORCH_MODEL_ZOO'] = '/vision/u/ldery/pytorchmodels'

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = self.make_layers(cfg16)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),  # nn.Dropout(), # TODO (ldery): Add dropout later
            # nn.Linear(4096, 4096),
            # nn.ReLU(True)  # nn.Dropout(), # TODO (ldery): Add dropout later
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def make_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
