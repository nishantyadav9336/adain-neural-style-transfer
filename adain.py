
import torch


def calc_mean_std(features, eps=1e-5):
    """
    Calculate channel-wise mean and standard deviation.
    """

    batch_size, channels = features.shape[:2]

    features = features.view(batch_size, channels, -1)

    mean = features.mean(dim=2).view(batch_size, channels, 1, 1)

    std = features.std(dim=2, unbiased=False).view(
        batch_size, channels, 1, 1
    )

    return mean, std + eps


def adaptive_instance_normalization(content, style):
    """
    Transfer style statistics to content features.
    """

    content_mean, content_std = calc_mean_std(content)
    style_mean, style_std = calc_mean_std(style)

    normalized_content = (content - content_mean) / content_std

    result = normalized_content * style_std + style_mean

    return result


# Test AdaIN
if __name__ == "__main__":

    content_features = torch.randn(1, 512, 32, 32)
    style_features = torch.randn(1, 512, 32, 32)

    output = adaptive_instance_normalization(
        content_features,
        style_features
    )

    print("Content shape:", content_features.shape)
    print("Style shape:", style_features.shape)
    print("AdaIN output shape:", output.shape)

    # Calculate statistics
    content_mean, content_std = calc_mean_std(content_features)
    style_mean, style_std = calc_mean_std(style_features)
    output_mean, output_std = calc_mean_std(output)

    print("\nContent mean:", content_mean.mean().item())
    print("Style mean:", style_mean.mean().item())
    print("Output mean:", output_mean.mean().item())

    print("\nStyle std:", style_std.mean().item())
    print("Output std:", output_std.mean().item())

    # Check whether AdaIN transferred the statistics
    print(
        "\nMean transferred:",
        torch.allclose(output_mean, style_mean, atol=1e-5)
    )

    print(
        "Std transferred:",
        torch.allclose(output_std, style_std, atol=1e-5)
    )
