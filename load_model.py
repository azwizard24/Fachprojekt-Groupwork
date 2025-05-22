import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import librosa
import numpy as np
import random
import os
import noisereduce as nr

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNNBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)  # Batch Normalization
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)  # Batch Normalization
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
    
    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x) 
        
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        x = self.relu(x)
        x = self.pool(x)
       
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_size, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(),
            nn.Linear(embed_size * 4, embed_size)
        )
        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.layer_norm2 = nn.LayerNorm(embed_size)
    
    def forward(self, x):
        # Attention
        attn_out, _ = self.self_attention(x, x, x)
        x = self.layer_norm1(x + attn_out)

        # Feed forward
        ffn_out = self.ffn(x)
        x = self.layer_norm2(x + ffn_out) 

        return x

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectionHead, self).__init__()
        self.proj = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.proj(x)

class BirdSoundModel(nn.Module):
    def __init__(self, input_channels, embed_size=128, num_heads=4, projection_dim=128):
        super(BirdSoundModel, self).__init__()
        
        # CNN blocks
        self.cnn_block1 = CNNBlock(input_channels, 32)
        self.cnn_block2 = CNNBlock(32, 64)
        #self.cnn_block3 = CNNBlock(64, 128)
        
        # Transformer Encoder Layers
        self.transformer_layer1 = TransformerEncoderLayer(embed_size=64, num_heads=num_heads)
        self.transformer_layer2 = TransformerEncoderLayer(embed_size=64, num_heads=num_heads)
        #self.transformer_layer3 = TransformerEncoderLayer(embed_size=128, num_heads=num_heads)
        
        # Projection head
        self.proj_head = ProjectionHead(input_dim=64, output_dim=projection_dim)

    def forward(self, x):
        # x shape: [batch_size, channels, height, width]
        # CNN layers
        x=x.float()
        x = self.cnn_block1(x)
        x = self.cnn_block2(x)
        #x = self.cnn_block3(x)

        x = x.flatten(start_dim=2).transpose(1,2)  # [batch_size, time_steps, features]
        
        # Transformer layers
        x = self.transformer_layer1(x)
        x = self.transformer_layer2(x)
        #x = self.transformer_layer3(x)
    
    
        # Projection head
        x = self.proj_head(x)
        x = x.mean(dim=1)
        return x

def preprocess_audio(file_path, n_fft=1024, hop_length=256, win_length=1024):
    y, sr = librosa.load(file_path, sr=None)
    y = nr.reduce_noise(y=y, sr=sr)
    y = librosa.effects.percussive(y) + librosa.effects.harmonic(y)
    y = y / (np.max(np.abs(y)) + 1e-6)
    return y, sr

model = BirdSoundModel(1, embed_size=128, num_heads=4, projection_dim=128)
file_1 =  "loop_test/21038/iNat65519_segment_2.wav"
file_2 = "loop_test/21038/iNat65519_segment_7.wav"
y1, sr = preprocess_audio(file_1, n_fft=1024, hop_length=256, win_length=1024)
#target_length = len(y)
def waveform_to_spectrogram(y, n_fft=1024, hop_length=256, win_length=1024):
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    return S_db
S1 = waveform_to_spectrogram(y1, n_fft=1024, hop_length=256, win_length=1024)
y2, sr = preprocess_audio(file_1, n_fft=1024, hop_length=256, win_length=1024)
#target_length = len(y)
S2 = waveform_to_spectrogram(y2, n_fft=1024, hop_length=256, win_length=1024)

model.load_state_dict(torch.load("feature_extractor.pth"))
# Move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(model)

model.eval()
S1 = torch.tensor(S1).float()
S1 = S1.unsqueeze(0).to(device)
S1 = S1.unsqueeze(1)
print(S1.shape)
S2 = torch.tensor(S2).float()
S2 = S2.unsqueeze(0).to(device)
S2 = S2.unsqueeze(1)
y1_result = model(S1)
y2_result = model(S2)
print(y1_result.cpu())
print(np.dot(y1_result.detach().cpu().numpy(), y2_result.detach().cpu().numpy().T))
# Example Forward Pass)
#input_data = torch.randn(32, 1, 513, 431)  # Example batch size of 32
#output = model(input_data)
#print(output.shape)  