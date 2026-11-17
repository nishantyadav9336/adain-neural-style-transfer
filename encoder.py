
import torch
import torch.nn as nn


class VGGEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(3, 3, 1),

            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 64, 3),
            nn.ReLU(),

            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),

            nn.MaxPool2d(2, 2, 0, ceil_mode=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),

            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),

            nn.MaxPool2d(2, 2, 0, ceil_mode=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),

            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),

            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),

            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),

            nn.MaxPool2d(2, 2, 0, ceil_mode=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 512, 3),
            nn.ReLU()
        )

        # Load pretrained VGG weights
        checkpoint = torch.load(
            "models/vgg_normalised.pth",
            map_location="cpu",
            weights_only=True
        )

        result = self.encoder.load_state_dict(
            checkpoint,
            strict=False
        )

        # print("Missing keys:", result.missing_keys)
        # print("Unexpected keys:", result.unexpected_keys)

        # Freeze encoder parameters
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        self.encoder.eval()

    def forward(self, image):
        return self.encoder(image)


if __name__ == "__main__":

    model = VGGEncoder()

    dummy_image = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        output = model(dummy_image)

    print("Pretrained VGG encoder loaded successfully!")
    print("Output feature shape:", output.shape)
