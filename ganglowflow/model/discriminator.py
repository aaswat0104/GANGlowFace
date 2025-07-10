import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels=3, img_size=64):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, 64, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),
            nn.Flatten(),
            nn.Linear(128 * (img_size // 4) ** 2, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)
