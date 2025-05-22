from new_dataset import ContrastiveAudioDataset, DistinctClassBatchSampler
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from model import BirdSoundModel

import torch
import torch.nn.functional as F
import torch
from torch import nn, optim

# NT-Xent loss function
def ntxent_loss_with_labels(out0, labels, temperature=0.5):
    out0 = nn.functional.normalize(out0, dim=1)

    # Calculate cosine similarity (pairwise similarity matrix)
    logits = torch.einsum("nc,mc->nm", out0, out0) / temperature

    # Mask diagonal (self-similarity)
    batch_size = out0.size(0)
    mask = torch.eye(batch_size, device=out0.device, dtype=torch.bool)
    logits = logits[~mask].view(batch_size, -1)  # Remove self-similarities

    # Generate positive pair labels: Same label is a positive pair
    labels = labels.unsqueeze(0) == labels.unsqueeze(1)  # Shape: (batch_size, batch_size)
    labels = labels.float()

    # Cross-entropy loss: maximize similarity for positive pairs, minimize for negative pairs
    cross_entropy = nn.CrossEntropyLoss(reduction="mean")

    # The target labels should be indices where labels are 1 (positive pairs)
    target = labels.argmax(dim=1)

    logits = logits.to("cuda:0")
    target = target.to("cuda:0")
    
    # Calculate the loss
    loss = cross_entropy(logits, target)

    return loss

# Set the device to CUDA if available, otherwise fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize dataset and sampler
directory = "/home/chandan/dmc/segment_audio"
batch_size = 128
dataset = ContrastiveAudioDataset(directory)
sampler = DistinctClassBatchSampler(dataset.labels, batch_size)
encodingDataLoader = DataLoader(dataset, batch_sampler=sampler)
print(len(encodingDataLoader))
# Initialize model
model = BirdSoundModel(input_channels=1, embed_size=128, num_heads=4, projection_dim=128).to(device)

# Total number of parameters in the model
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

# Set up the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Training loop
num_epochs = 15  # Set the number of epochs you want to train for

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    batch_count = 0
    running_loss = 0.0
    for s1, label in encodingDataLoader:
        s1 = s1.to(device)  # Add a channel dimension to the input
        label = label.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        out1 = model(s1)

        # Calculate loss
        loss = ntxent_loss_with_labels(out1, label, temperature=0.5)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate the loss
        running_loss += loss.item()
        batch_count += 1
        
        if batch_count % 2 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_count}], Loss: {running_loss / 2:.4f}")
            running_loss = 0.0
    # Print loss for the current epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(encodingDataLoader)}")

    # Optional: Save the model checkpoint every epoch
    torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")

# Final model save after training is complete
torch.save(model.state_dict(), "final_model.pth")
