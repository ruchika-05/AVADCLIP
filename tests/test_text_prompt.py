
import torch

from models.visual_encoder import VisualEncoder
from models.text_prompt import LearnableTextPrompt

# Load CLIP
encoder = VisualEncoder()

# Define anomaly categories
class_names = [
    "normal activity",
    "fighting",
    "shooting",
    "explosion",
    "car accident",
    "abuse",
    "riot"
]

# Create learnable prompts
prompt_model = LearnableTextPrompt(
    clip_model=encoder.model,
    class_names=class_names,
    n_ctx=4
)

prompt_model.train()

text_features = prompt_model()
loss = text_features.sum()
loss.backward()

print("Text prompt feature shape:", text_features.shape)
print("Prompt gradients available:", prompt_model.context.grad is not None)
print("CLIP frozen:", all(
    not p.requires_grad for p in encoder.model.parameters()
))
