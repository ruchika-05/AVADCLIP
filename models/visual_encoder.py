
import torch
import open_clip


class VisualEncoder:
    def __init__(self, device=None):

        # Select Apple GPU if available
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)

        print("Loading CLIP ViT-B/16...")

        # Load pretrained CLIP
        self.model, _, self.preprocess = (
            open_clip.create_model_and_transforms(
                "ViT-B-16",
                pretrained="openai"
            )
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        # Freeze pretrained parameters
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        print("CLIP loaded successfully!")
        print("Device:", self.device)

    def extract_features(self, image_paths):

        images = []

        # Load and preprocess images
        from PIL import Image

        for path in image_paths:
            image = Image.open(path).convert("RGB")
            image = self.preprocess(image)
            images.append(image)

        images = torch.stack(images).to(self.device)

        # Extract embeddings
        with torch.inference_mode():
            features = self.model.encode_image(images)

            # L2 normalization
            features = features / features.norm(
                dim=-1, keepdim=True
            )

        return features.cpu()
    def encode_text(self, texts):

        # Convert text into CLIP tokens
        tokens = open_clip.tokenize(texts).to(self.device)

        # Extract text embeddings
        with torch.no_grad():
            features = self.model.encode_text(tokens)

            # L2 normalization
            features = features / features.norm(
                dim=-1, keepdim=True
            )

        return features.cpu()
