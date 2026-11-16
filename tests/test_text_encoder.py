from models.visual_encoder import VisualEncoder

encoder = VisualEncoder()

texts = [
    "A normal video showing people walking",
    "A video showing people fighting",
    "A video showing an explosion",
    "A video showing a person shooting"
]

features = encoder.encode_text(texts)

print("Text feature shape:", features.shape)
print("Text encoding completed!")
print("\nCLIP text encoder details:")
print("Token embedding:", encoder.model.token_embedding.weight.shape)
print("Positional embedding:", encoder.model.positional_embedding.shape)
print("Text projection:", encoder.model.text_projection.shape)