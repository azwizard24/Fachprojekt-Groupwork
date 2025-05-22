import json
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
from torchaudio.transforms import Resample

class MultiLabelAudioDataset(Dataset):
    def __init__(self, annotation_file, sample_rate=32000):
        """
        Args:
            annotation_file (str): Path to JSON file with {file_path: [label_ids]} mapping.
            sample_rate (int): Target audio sample rate.
        """
        with open(annotation_file) as f:
            self.annotations = json.load(f)
        self.annotations = {key:value for annotation in self.annotations for key,value in annotation.items()}
        self.sample_rate = sample_rate
        self.file_paths = list(self.annotations.keys())
        self.num_classes = max(max(labels) for labels in self.annotations.values()) + 1

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        labels = self.annotations[file_path]

        waveform, sr = torchaudio.load(file_path)
        if sr != self.sample_rate:
            waveform = Resample(sr, self.sample_rate)(waveform)
        waveform = waveform.mean(dim=0)  # mono

        label_tensor = torch.zeros(self.num_classes)
        label_tensor[labels] = 1.0

        return {"audio": waveform, "label": label_tensor}

def load_dataloaders(annotation_file, batch_size=32, sample_rate=32000, train_split=0.8):
    dataset = MultiLabelAudioDataset(annotation_file, sample_rate=sample_rate)
    train_size = int(len(dataset) * train_split)
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, dataset.num_classes