import torch
import torchaudio
import json
import os
from bird_model import BirdsetModule
from torchaudio.transforms import Resample

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
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0)

def infer_multi_label(model, audio_tensor, device='cpu', threshold=0.8):
    model.eval()
    with torch.no_grad():
        audio_tensor = audio_tensor.to(device)
        logits = model(audio_tensor.unsqueeze(0))
        probs = torch.sigmoid(logits)
        predicted_indices = (probs > threshold).nonzero(as_tuple=True)[1].tolist()
        predicted_scores = probs[0, predicted_indices].tolist()
        return predicted_indices, predicted_scores

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "checkpoints/birdset_epoch1.pt"
    audio_dir = "segment_audio_ss"
    output_json = "multi_label_predictions.json"
    model = load_model(checkpoint_path, device=device)
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
    results = []

    for idx, audio_file in enumerate(audio_files):
        audio_path = os.path.join(audio_dir, audio_file)
        audio_tensor = preprocess_audio(audio_path)
        labels, scores = infer_multi_label(model, audio_tensor, device=device)
        results.append({ audio_path: labels})
        print(f"Done {audio_path}")
        
        if idx%250 == 0:
            print(idx)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Predictions written to {output_json}")

if __name__ == "__main__":
    main()
