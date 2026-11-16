import torch

from models.avadclip_model import AVadCLIPModel


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    batch_size, num_segments, feature_dim, num_classes = 2, 16, 512, 7
    model = AVadCLIPModel(feature_dim=feature_dim).to(device)
    visual = torch.randn(
        batch_size, num_segments, feature_dim, device=device
    )
    audio = torch.randn_like(visual)
    class_features = torch.randn(
        num_classes, feature_dim, device=device
    )

    output = model(visual, audio, class_features=class_features)

    assert output["segment_scores"].shape == (batch_size, num_segments)
    assert torch.all(
        (output["segment_scores"] >= 0)
        & (output["segment_scores"] <= 1)
    )
    assert output["video_score"].shape == (batch_size,)
    assert output["fused_features"].shape == (
        batch_size, num_segments, feature_dim
    )
    assert output["fusion_weights"].shape == (
        batch_size, num_segments, feature_dim
    )
    assert output["enhanced_class_features"].shape == (
        batch_size, num_classes, feature_dim
    )
    assert output["alignment_map"].shape == (
        batch_size, num_classes, num_segments
    )
    assert output["class_scores"].shape == (batch_size, num_classes)
    assert output["class_probabilities"].shape == (
        batch_size, num_classes
    )
    expected_k = 2
    expected_video_scores = torch.topk(
        output["segment_scores"],
        k=expected_k,
        dim=1,
    ).values.mean(dim=1)
    assert torch.allclose(output["video_score"], expected_video_scores)

    loss = output["video_score"].mean() + output["class_scores"].mean()
    loss.backward()
    assert model.anomaly_detector.feature_mlp[0].weight.grad is not None
    assert model.fusion.weight_layer.weight.grad is not None
    assert model.audio_visual_prompt.prompt_ffn[0].weight.grad is not None

    print("Segment scores:", output["segment_scores"].shape)
    print("Video score:", output["video_score"].shape)
    print("Fused features:", output["fused_features"].shape)
    print("Fusion weights:", output["fusion_weights"].shape)
    print("Enhanced class features:", output["enhanced_class_features"].shape)
    print("Alignment map:", output["alignment_map"].shape)
    print("Class probabilities:", output["class_probabilities"].shape)
    print("\nModel integration test successful!")


if __name__ == "__main__":
    main()
