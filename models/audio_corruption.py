import numpy as np


SAMPLE_RATE = 16000
FRAME_LENGTH = 16000
HOP_LENGTH = 16000

CORRUPTION_CONDITIONS = (
    "volume_0_5",
    "volume_0_1",
    "noise_snr_20",
    "noise_snr_10",
    "noise_snr_0",
    "partial_mute_50",
    "silence",
)


def _frame_count(sample_count):
    return max(1, 1 + max(0, sample_count - FRAME_LENGTH) // HOP_LENGTH)


def corrupt_waveform(waveform, condition, seed):
    if condition not in CORRUPTION_CONDITIONS:
        raise ValueError(f"Unknown audio corruption condition: {condition}")
    audio = np.asarray(waveform, dtype=np.float32).copy()
    if audio.ndim != 1:
        raise ValueError("Expected a mono waveform.")

    frames = _frame_count(len(audio))
    targets = np.ones(frames, dtype=np.float32)
    rng = np.random.default_rng(seed)

    if condition == "volume_0_5":
        audio *= 0.5
    elif condition == "volume_0_1":
        audio *= 0.1
    elif condition.startswith("noise_snr_"):
        snr = float(condition.rsplit("_", 1)[1])
        signal_rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
        signal_rms = max(signal_rms, 1e-4)
        noise_rms = signal_rms * (10.0 ** (-snr / 20.0))
        noise = rng.normal(0.0, 1.0, size=audio.shape).astype(np.float32)
        noise_rms_actual = float(
            np.sqrt(np.mean(np.square(noise, dtype=np.float64)))
        )
        audio += noise * (noise_rms / noise_rms_actual)
    elif condition == "partial_mute_50":
        muted_frames = max(1, frames // 2)
        start_frame = int(rng.integers(0, frames - muted_frames + 1))
        start_sample = start_frame * HOP_LENGTH
        end_sample = min(
            len(audio),
            (start_frame + muted_frames) * HOP_LENGTH,
        )
        audio[start_sample:end_sample] = 0.0
        targets[start_frame:start_frame + muted_frames] = 0.0
    elif condition == "silence":
        audio.fill(0.0)
        targets.fill(0.0)

    return audio, targets
