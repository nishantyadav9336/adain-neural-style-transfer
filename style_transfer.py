
import torch
from PIL import Image
from torchvision import transforms

from encoder import VGGEncoder
from decoder import Decoder
from adain import adaptive_instance_normalization


# Load pretrained models
def load_models():

    device = torch.device("cpu")

    encoder = VGGEncoder().to(device)
    decoder = Decoder().to(device)

    checkpoint = torch.load(
        "models/decoder.pth",
        map_location=device,
        weights_only=True
    )

    decoder.decoder.load_state_dict(checkpoint)

    encoder.eval()
    decoder.eval()

    return encoder, decoder


# Generate stylized image
def generate_stylized_image(
    content_image,
    style_image,
    alpha,
    resolution,
    encoder,
    decoder
):
    transform = transforms.Compose([
        transforms.Resize(resolution),
        transforms.ToTensor()
    ])

    content_tensor = transform(
        content_image.convert("RGB")
    ).unsqueeze(0)

    style_tensor = transform(
        style_image.convert("RGB")
    ).unsqueeze(0)

    with torch.no_grad():

        content_features = encoder(content_tensor)
        style_features = encoder(style_tensor)

        adain_features = adaptive_instance_normalization(
            content_features,
            style_features
        )

        target_features = (
            alpha * adain_features
            + (1 - alpha) * content_features
        )

        output = decoder(target_features)

    output = output.squeeze(0).cpu()

    # Remove extra pixels added by the decoder
    height, width = content_tensor.shape[2:]
    output = output[:, :height, :width]

    output = output.clamp(0, 1)

    output_image = transforms.ToPILImage()(output)

    return output_image
