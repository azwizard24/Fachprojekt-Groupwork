import torch
import torchaudio
#from birdset import ConvNextBirdSet
from bird_model import BirdsetModule  # assuming this is the training script filename
from torchaudio.transforms import Resample
import os

def load_model(checkpoint_path, num_classes=206, device='cpu'):
    model = BirdsetModule(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def preprocess_audio(audio_path, target_sample_rate=32000):
    waveform, sample_rate = torchaudio.load(audio_path)

    if sample_rate != target_sample_rate:
        resampler = Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    # shape: [1, time] -> remove channel if needed
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # convert to mono
    return waveform.squeeze(0)  # [time]

def infer_single_audio(model, audio_tensor, device='cpu', top_k=5):
    model.eval()
    with torch.no_grad():
        audio_tensor = audio_tensor.to(device)
        logits = model(audio_tensor.unsqueeze(0))  # add batch dim
        probs = torch.softmax(logits, dim=1)
        top_probs, top_indices = torch.topk(probs, k=top_k)
        return top_indices.squeeze(0).tolist(), top_probs.squeeze(0).tolist()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = "checkpoints/birdset_epoch3.pt"  # change this to your actual checkpoint
    audio_path = "/home/chandan/dmc/segment_audio/ywcpar/iNat33510_segment_3.wav"  # replace with your audio file

    model = load_model(checkpoint_path, device=device)
    audio_tensor = preprocess_audio(audio_path)
    top_classes, top_scores = infer_single_audio(model, audio_tensor, device=device)

    print(f"Top Predictions:")
    for idx, score in zip(top_classes, top_scores):
        print(f"Class ID: {idx}, Confidence: {score:.4f}")

if __name__ == "__main__":
    main()
