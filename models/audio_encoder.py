
import torch
import librosa
import wav2clip
import numpy as np
from pathlib import Path


class AudioEncoder:

    def __init__(self):
        print("Loading Wav2CLIP...")

        # Store the model inside our project folder
        project_dir = Path(__file__).resolve().parents[1]
        cache_dir = project_dir / ".cache" / "torch_hub"

        cache_dir.mkdir(parents=True, exist_ok=True)

        # Set PyTorch's model download directory
        torch.hub.set_dir(str(cache_dir))

        self.model = wav2clip.get_model(
            frame_length=16000,
            hop_length=16000
        )

        print("Wav2CLIP loaded successfully!")

    def extract_features(self, audio_path):

        audio, sample_rate = librosa.load(
            audio_path,
            sr=16000,
            mono=True
        )

        features = wav2clip.embed_audio(audio, self.model)

        features = np.asarray(features)

        return torch.tensor(features, dtype=torch.float32)
