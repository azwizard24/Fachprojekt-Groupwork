import os
from torch.utils.data import Dataset, Sampler
from collections import defaultdict
import librosa
import numpy as np
import noisereduce as nr
import random
import torchaudio.transforms as T
import torch

def preprocess_audio(file_path, n_fft=1024, hop_length=256, win_length=1024):
    y, sr = librosa.load(file_path, sr=None)
    y = nr.reduce_noise(y=y, sr=sr)
    y = librosa.effects.percussive(y) + librosa.effects.harmonic(y)
    y = y / (np.max(np.abs(y)) + 1e-6)
    return y, sr

def waveform_to_spectrogram(y, n_fft=1024, hop_length=256, win_length=1024):
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    return S_db

class DistinctClassBatchSampler(Sampler):
    def __init__(self, labels, batch_size):
        assert batch_size % 2 == 0, "Batch size must be even (2 samples per class)"
        self.labels = labels
        self.batch_size = batch_size
        self.samples_per_class = 2  # Ensure 2 samples per class per batch
        self.num_samples = len(labels)

        # Map class -> list of indices
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)
        
        # Flatten the label_to_indices dictionary into a list of indices
        self.all_indices = []
        for label, indices in self.label_to_indices.items():
            self.all_indices.extend(indices)

    def __iter__(self):
        # Shuffle the indices to ensure randomness
        random.shuffle(self.all_indices)

        # Iterate over the indices and generate batches
        batch = []
        for idx in range(0, len(self.all_indices), self.batch_size):
            batch.clear()

            # We need to form the batch by taking 2 samples from the same class
            while len(batch) < self.batch_size:
                # Pick a random class from the available classes
                random_class = random.choice(list(self.label_to_indices.keys()))
                # Get 2 samples for that class
                samples = random.sample(self.label_to_indices[random_class], self.samples_per_class)
                batch.extend(samples)

            yield batch
    def __len__(self):
        # Calculate number of batches: Total samples divided by batch_size
        return len(self.all_indices) // self.batch_size

class ContrastiveAudioDataset(Dataset):
    def __init__(self, root_dir, n_fft=1024, hop_length=256, win_length=1024, n_mels=128, target_sample_rate=16000):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.target_sample_rate = target_sample_rate

        self.samples = []
        self.labels = []
        self.label_to_indices = defaultdict(list)

        current_label = 0
        self.class_to_label = {}
        self.label_to_class = {}

        for class_name in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            self.class_to_label[class_name] = current_label
            self.label_to_class[current_label] = class_name

            for fname in os.listdir(class_path):
                full_path = os.path.join(class_path, fname)
                self.samples.append(full_path)
                self.labels.append(current_label)
                self.label_to_indices[current_label].append(len(self.samples) - 1)

            current_label += 1

        # Mel + Log transform setup once
        self.mel_transform = torch.nn.Sequential(
            T.MelSpectrogram(
                sample_rate=self.target_sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_mels=self.n_mels
            ),
            T.AmplitudeToDB()
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        label = self.labels[idx]

        # Preprocess waveform
        y, sr = preprocess_audio(path, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)

        # Resample if needed
        if sr != self.target_sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sample_rate)
            sr = self.target_sample_rate

        # Convert to tensor: [1, time]
        waveform = torch.tensor(y, dtype=torch.float32).unsqueeze(0)

        # Apply mel transform
        mel_spec = self.mel_transform(waveform)  # [1, n_mels, time]

        return mel_spec, label