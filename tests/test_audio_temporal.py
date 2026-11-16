import torch

from models.audio_temporal_model import AudioTemporalEncoder


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    feature_dim = 512
    model = AudioTemporalEncoder(feature_dim=feature_dim).to(device)
    model.train()

    video_features = torch.randn(
        2, 33, feature_dim, device=device, requires_grad=True
    )
    output = model(video_features)
    assert output.shape == video_features.shape
    assert torch.isfinite(output).all()

    output.square().mean().backward()
    assert video_features.grad is not None
    block_parameters = [
        parameter
        for parameter in model.blocks[0].parameters()
        if parameter.requires_grad
    ]
    assert block_parameters
    assert all(parameter.grad is not None for parameter in block_parameters)

    single_video = torch.randn(17, feature_dim, device=device)
    single_output = model(single_video)
    assert single_output.shape == (1, 17, feature_dim)

    print("Batched input:", video_features.shape)
    print("Batched output:", output.shape)
    print("Single-video output:", single_output.shape)
    print("\nAudio temporal TCN test successful!")


if __name__ == "__main__":
    main()
