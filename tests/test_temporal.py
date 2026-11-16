import torch

from models.temporal_model import VisualTemporalEncoder


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = VisualTemporalEncoder().to(device)
    model.train()

    visual = torch.randn(2, 33, 512, device=device, requires_grad=True)
    output = model(visual)
    assert output.shape == visual.shape
    output.square().mean().backward()
    assert model.similarity_projection.grad is not None
    assert model.output_projection.weight.grad is not None
    assert model.similarity_gc1.weight.grad is not None

    model.eval()
    with torch.no_grad():
        long_visual = torch.randn(1, 683, 512, device=device)
        long_output = model(long_visual)
    assert long_output.shape == long_visual.shape
    assert torch.isfinite(long_output).all()

    print("Device:", device)
    print("Batched input/output:", visual.shape, output.shape)
    print("Longest dataset input/output:", long_visual.shape, long_output.shape)
    print("Visual temporal model test successful!")


if __name__ == "__main__":
    main()
