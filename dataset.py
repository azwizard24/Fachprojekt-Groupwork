import librosa
import numpy as np
import random
import os
import noisereduce as nr

# Preprocess original audio
def preprocess_audio(file_path, n_fft=1024, hop_length=256, win_length=1024):
    y, sr = librosa.load(file_path, sr=None)
    y = nr.reduce_noise(y=y, sr=sr)
    y = librosa.effects.percussive(y) + librosa.effects.harmonic(y)
    y = y / (np.max(np.abs(y)) + 1e-6)
    return y, sr

# Match length for augmentations
def match_length(y_augmented, target_length):
    if len(y_augmented) > target_length:
        y_augmented = y_augmented[:target_length]
    elif len(y_augmented) < target_length:
        padding = target_length - len(y_augmented)
        y_augmented = np.pad(y_augmented, (0, padding))
    return y_augmented

# Augment waveform (positive sample)
def augment_waveform(y, sr):
    augmentations = [
        lambda y: librosa.effects.time_stretch(y, rate=random.uniform(0.8, 1.2)),
        lambda y: librosa.effects.pitch_shift(y, sr=sr, n_steps=random.randint(-3, 3)),
        lambda y: y + np.random.normal(0, random.uniform(0.002, 0.01), size=y.shape),
        lambda y: librosa.effects.preemphasis(y, coef=random.uniform(0.97, 0.98)),
    ]
    aug = random.choice(augmentations)
    return aug(y)

# Convert waveform to spectrogram
def waveform_to_spectrogram(y, n_fft=1024, hop_length=256, win_length=1024):
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    return S_db

class ContrastiveDataset:
    def __init__(self, dataset_files, n_fft=1024, hop_length=256, win_length=1024):
        self.dataset_files = dataset_files
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def __getitem__(self, idx):
        file_path = self.dataset_files[idx]
        
        # Load and preprocess audio
        y, sr = preprocess_audio(file_path, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
        target_length = len(y)

        # Create two augmented views
        y1 = augment_waveform(y, sr)
        y2 = augment_waveform(y, sr)

        # Match length if augmentation changed it
        y1 = match_length(y1, target_length)
        y2 = match_length(y2, target_length)

        # Convert to spectrogram
        S1 = waveform_to_spectrogram(y1, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
        S2 = waveform_to_spectrogram(y2, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)

        #directory name as class label
        label = os.path.basename(os.path.dirname(file_path))
        #returning original only for viz
        #y1 = waveform_to_spectrogram(y)
        return S1, S2, label

    def __len__(self):
        return len(self.dataset_files)
