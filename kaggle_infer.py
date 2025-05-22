from torch import nn
import torch
import torchaudio
from torchaudio.transforms import Resample
from torchvision import transforms
import datasets
import warnings
import pandas as pd
import json
import os

from typing import Dict, Optional

import datasets
import torch
from torch import nn
from transformers import AutoConfig, ConvNextForImageClassification

class PowerToDB(nn.Module):
    def __init__(self, ref=1.0, amin=1e-10, top_db=80.0):
        super(PowerToDB, self).__init__()
        # Initialize parameters
        self.ref = ref
        self.amin = amin
        self.top_db = top_db

    def forward(self, S):
        # Convert S to a PyTorch tensor if it is not already
        S = torch.as_tensor(S, dtype=torch.float32)

        if self.amin <= 0:
            raise ValueError("amin must be strictly positive")

        if torch.is_complex(S):
            warnings.warn(
                "power_to_db was called on complex input so phase "
                "information will be discarded. To suppress this warning, "
                "call power_to_db(S.abs()**2) instead.",
                stacklevel=2,
            )
            magnitude = S.abs()
        else:
            magnitude = S

        # Check if ref is a callable function or a scalar
        if callable(self.ref):
            ref_value = self.ref(magnitude)
        else:
            ref_value = torch.abs(torch.tensor(self.ref, dtype=S.dtype))

        # Compute the log spectrogram
        log_spec = 10.0 * torch.log10(
            torch.maximum(magnitude, torch.tensor(self.amin, device=magnitude.device))
        )
        log_spec -= 10.0 * torch.log10(
            torch.maximum(ref_value, torch.tensor(self.amin, device=magnitude.device))
        )

        # Apply top_db threshold if necessary
        if self.top_db is not None:
            if self.top_db < 0:
                raise ValueError("top_db must be non-negative")
            log_spec = torch.maximum(log_spec, log_spec.max() - self.top_db)

        return log_spec

class ConvNextClassifier(nn.Module):
    """
    ConvNext model for audio classification.
    """

    def __init__(
        self,
        num_channels: int = 1,
        num_classes: Optional[int] = None,
        checkpoint: Optional[str] = None,
        local_checkpoint: Optional[str] = None,
        cache_dir: Optional[str] = None,
        pretrain_info= None #PretrainInfoConfig = None,
    ):
        """
        Note: Either num_classes or pretrain_info must be given
        Args:
            num_channels: Number of input channels.
            checkpoint: huggingface checkpoint path of any model of correct type
            num_classes: number of classification heads to be used in the model
            local_checkpoint: local path to checkpoint file
            cache_dir: specified cache dir to save model files at
            pretrain_info: hf_path and hf_name of info will be used to infer if num_classes is None
        """
        super().__init__()

        if pretrain_info:
            self.hf_path = pretrain_info.hf_path
            self.hf_name = (
                pretrain_info.hf_name
                if not pretrain_info.hf_pretrain_name
                else pretrain_info.hf_pretrain_name
            )
            self.num_classes = len(
                datasets.load_dataset_builder(self.hf_path, self.hf_name)
                .info.features["ebird_code"]
                .names
            )
        else:
            self.hf_path = None
            self.hf_name = None
            self.num_classes = num_classes

        self.num_channels = num_channels
        self.checkpoint = checkpoint
        self.local_checkpoint = local_checkpoint
        self.cache_dir = cache_dir

        self.model = None

        self._initialize_model()

    def _initialize_model(self):
        """Initializes the ConvNext model based on specified attributes."""

        adjusted_state_dict = None

        if self.checkpoint:
            if self.local_checkpoint:
                state_dict = torch.load(self.local_checkpoint)["state_dict"]

                # Update this part to handle the necessary key replacements
                adjusted_state_dict = {}
                for key, value in state_dict.items():
                    # Handle 'model.model.' prefix
                    new_key = key.replace("model.model.", "")

                    # Handle 'model._orig_mod.model.' prefix
                    new_key = new_key.replace("model._orig_mod.model.", "")

                    # Assign the adjusted key
                    adjusted_state_dict[new_key] = value

            self.model = ConvNextForImageClassification.from_pretrained(
                self.checkpoint,
                num_labels=self.num_classes,
                num_channels=self.num_channels,
                cache_dir=self.cache_dir,
                state_dict=adjusted_state_dict,
                ignore_mismatched_sizes=True,
            )
        else:
            config = AutoConfig.from_pretrained(
                "facebook/convnext-base-224-22k",
                num_labels=self.num_classes,
                num_channels=self.num_channels,
            )
            self.model = ConvNextForImageClassification(config)

    def forward(
        self, input_values: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Defines the forward pass of the ConvNext model.

        Args:
            input_values (torch.Tensor): An input batch.
            labels (Optional[torch.Tensor]): The corresponding labels. Default is None.

        Returns:
            torch.Tensor: The output of the ConvNext model.
        """
        output = self.model(input_values)
        logits = output.logits

        return logits

    @torch.inference_mode()
    def get_logits(self, dataloader, device):
        pass

    @torch.inference_mode()
    def get_probas(self, dataloader, device):
        pass

    @torch.inference_mode()
    def get_representations(self, dataloader, device):
        pass

class ConvNextBirdSet(nn.Module):
    """
    BirdSet ConvNext model trained on BirdSet XCL dataset.
    The model expects a raw 1 channel 5s waveform with sample rate of 32kHz as an input.
    Its preprocess function will:
        - convert the waveform to a spectrogram: n_fft: 1024, hop_length: 320, power: 2.0
        - melscale the spectrogram: n_mels: 128, n_stft: 513
        - dbscale with top_db: 80
        - normalize the spectrogram mean: -4.268, std: 4.569 (from esc-50)
    """

    def __init__(
        self,
        PowerToDB,
        num_classes=9736,
    ):
        super().__init__()
        self.model = ConvNextClassifier(
            checkpoint="DBD-research-group/ConvNeXT-Base-BirdSet-XCL",
            num_classes=num_classes,
        )
        self.spectrogram_converter = torchaudio.transforms.Spectrogram(
            n_fft=1024, hop_length=320, power=2.0
        )
        self.mel_converter = torchaudio.transforms.MelScale(
            n_mels=128, n_stft=513, sample_rate=32_000
        )
        self.normalizer = transforms.Normalize((-4.268,), (4.569,))
        self.powerToDB = PowerToDB(top_db=80)
        self.config = self.model.model.config

    def preprocess(self, waveform: torch.Tensor):
        # convert waveform to spectrogram
        spectrogram = self.spectrogram_converter(waveform)
        spectrogram = spectrogram.to(torch.float32)
        melspec = self.mel_converter(spectrogram)
        dbscale = self.powerToDB(melspec)
        normalized_dbscale = self.normalizer(dbscale)
        # add dimension 3 from left
        normalized_dbscale = normalized_dbscale.unsqueeze(-3)

        return normalized_dbscale

    def forward(self, input: torch.Tensor):
        # spectrogram = self.preprocess(waveform)
        return self.model(input)


def preprocess_audio(audio_path, target_sample_rate=32000):
    waveform, sample_rate = torchaudio.load(audio_path, backend="soundfile")

    if sample_rate != target_sample_rate:
        resampler = Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    # shape: [1, time] -> remove channel
        waveform = waveform.mean(dim=0, keepdim=True)  # convert to mono
    return waveform.squeeze(0)  # [time]


class BirdsetModule(torch.nn.Module):
    def __init__(self,PowerToDB,num_classes=206):
        super().__init__()
        self.model = ConvNextBirdSet(PowerToDB, num_classes=num_classes)

    def forward(self, x):
        preprocessed = self.model.preprocess(x)
        return self.model(preprocessed)
    
def inference(model, audio_tensor, device):
    model.eval()
    with torch.no_grad():
        audio_tensor = audio_tensor.to(device)
        logits = model(audio_tensor.unsqueeze(0))  # add batch dim
        probs = torch.softmax(logits, dim=1)
        #top_probs, top_indices = torch.topk(probs, k=top_k)
        return probs.squeeze(0).tolist() #top_indices.squeeze(0).tolist(), top_probs.squeeze(0).tolist()

checkpoint_path = "./checkpoints/birdset_epoch3.pt"

device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BirdsetModule(PowerToDB, num_classes=206).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
test_dir = "train_ss"
with open("map.txt","r") as file:
    content = file.read()
map_text = json.loads(content)

probabilities=[]
for file_path in os.listdir(test_dir):
    audio_path = os.path.join(test_dir,file_path)
    audio_tensor = preprocess_audio(audio_path)
    probs = inference(model, audio_tensor,device)
    probabilities.append(probs)

columns = list(map_text.values())
df = pd.DataFrame(probabilities,columns=columns)
df.to_csv("submission.csv")