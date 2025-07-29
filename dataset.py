import os
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from torchaudio.transforms import Resample

class CustomAudioDataset(Dataset):
    def __init__(self, root_dir, transform=None, sample_rate=32000):
        """
        Args:
            root_dir (string): Directory with subfolders of audio files.
            transform (callable, optional): Optional transform on sample.
            sample_rate (int): Resampling frequency.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.sample_rate = sample_rate
        self.audio_files = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        self._prepare_data()

    def _prepare_data(self):
        classes = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}

        for class_name, idx in self.class_to_idx.items():
            class_dir = os.path.join(self.root_dir, class_name)
            for audio_file in os.listdir(class_dir):
                if audio_file.endswith((".wav", ".mp3")):
                    file_path = os.path.join(class_dir, audio_file)
                    self.audio_files.append((file_path, idx))

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path, label = self.audio_files[idx]

        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != self.sample_rate:
            resampler = Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)
        waveform = waveform.squeeze(0)

        sample = {"audio": waveform, "label": label}
        if self.transform:
            sample = self.transform(sample)

        return sample

def load_dataloaders(root_dir, batch_size=32, sample_rate=32000, train_split=0.8): # Run with 0.8.
    dataset = CustomAudioDataset(root_dir=root_dir, sample_rate=sample_rate)
    print(dataset.idx_to_class)
    train_size = int(len(dataset) * train_split)
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

#Segment_audio should be the dataset with 5 sec clipped audio file.
root_dir = 'segment_audio'
train_loader, test_loader = load_dataloaders(root_dir)
for batch in train_loader:
    print(batch["audio"].shape)  # (batch_size, audio_length)
    print(batch["label"])
    break
