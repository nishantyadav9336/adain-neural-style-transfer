
import torch
import torch.nn as nn


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(

            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, 3),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode="nearest"),

            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),

            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),

            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),

            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, 3),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode="nearest"),

            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),

            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, 3),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode="nearest"),

            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),

            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, 3)
        )

    def forward(self, features):
        return self.decoder(features)


if __name__ == "__main__":

    model = Decoder()

    dummy_features = torch.randn(1, 512, 32, 32)

    with torch.no_grad():
        output = model(dummy_features)

    print("Decoder architecture loaded!")
    print("Input shape:", dummy_features.shape)
    print("Output shape:", output.shape)
