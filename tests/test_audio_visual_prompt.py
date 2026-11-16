import torch
from models.audio_visual_prompt import AudioVisualPrompt


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    batch_size, num_segments, feature_dim, num_classes = 2, 16, 512, 7
    av_features = torch.randn(
        batch_size, num_segments, feature_dim, device=device
    )
    anomaly_logits = torch.randn(
        batch_size, num_segments, device=device
    )
    text_features = torch.randn(
        num_classes, feature_dim, device=device, requires_grad=True
    )

    av_prompt = AudioVisualPrompt(feature_dim=feature_dim).to(device)

    prompt_output = av_prompt(
        av_features,
        anomaly_logits,
        text_features
    )

    enhanced_features = prompt_output["enhanced_features"]

    print("AV features shape:", av_features.shape)
    print("Anomaly logits shape:", anomaly_logits.shape)
    print("Text features shape:", text_features.shape)
    print("Enhanced text features shape:", enhanced_features.shape)
    print("Global features shape:", prompt_output["global_features"].shape)
    print("Class weights shape:", prompt_output["class_weights"].shape)

    assert enhanced_features.shape == (batch_size, num_classes, feature_dim)
    assert torch.allclose(
        prompt_output["multimodal_prompts"][:, :1],
        prompt_output["multimodal_prompts"][:, 1:]
    )
    assert torch.allclose(
        prompt_output["class_weights"],
        torch.full_like(
            prompt_output["class_weights"],
            1 / num_classes
        )
    )
    expected_global = torch.bmm(
        anomaly_logits.unsqueeze(1),
        av_features,
    ).squeeze(1)
    expected_global = torch.nn.functional.normalize(
        expected_global, dim=-1
    )
    assert torch.allclose(
        prompt_output["global_features"], expected_global
    )
    expected_prompt_input = (
        text_features.unsqueeze(0)
        + prompt_output["multimodal_prompts"]
    )
    expected_enhanced = (
        expected_prompt_input + av_prompt.prompt_ffn(expected_prompt_input)
    )
    assert torch.allclose(enhanced_features, expected_enhanced)

    loss = enhanced_features.mean()
    loss.backward()

    print(
        "Prompt FFN gradients available:",
        av_prompt.prompt_ffn[0].weight.grad is not None
    )
    print(
        "Class feature gradients available:",
        text_features.grad is not None
    )

    print("\nAudio-Visual Prompt test successful!")


if __name__ == "__main__":
    main()