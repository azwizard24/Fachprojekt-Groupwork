from new_dataset import ContrastiveAudioDataset, DistinctClassBatchSampler
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from model import BirdSoundModel

import torch
import torch.nn.functional as F
import torch
from torch import nn
def nt_xent_loss_three(anchor, positive, negative, temperature=0.5):
    anchor = F.normalize(anchor, p=2, dim=-1)
    positive = F.normalize(positive, p=2, dim=-1)
    negative = F.normalize(negative, p=2, dim=-1)
    
    print(anchor.shape)
    print(anchor.unsqueeze(1).shape, positive.unsqueeze(2).shape)
    sim_ap = torch.matmul(anchor.unsqueeze(1), positive.unsqueeze(2)).squeeze(-1)
    sim_an = torch.matmul(anchor.unsqueeze(1), negative.unsqueeze(2)).squeeze(-1)

    sim_ap = sim_ap / temperature
    sim_an = sim_an / temperature

    positive_term = torch.exp(sim_ap)
    negative_term = torch.exp(sim_an)

    denominator = positive_term + negative_term

    loss = -torch.log(positive_term / denominator)

    return loss.mean()
# NT-Xent loss function
def nt_xent_loss(z1, z2, temperature=0.5):
    """
    NT-Xent loss for two augmented views.
    """
    batch_size = z1.shape[0]
    
    # Normalize the embeddings
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Concatenate
    z = torch.cat([z1, z2], dim=0)  # (2N, D)
    
    # Similarity matrix
    sim_matrix = torch.matmul(z, z.T)  # (2N, 2N)
    sim_matrix = sim_matrix / temperature

    # Remove similarity of samples to themselves
    mask = torch.eye(sim_matrix.size(0), device=z.device).bool()
    sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))

    # Positive pairs (i, i+N) and (i+N, i)
    positives = torch.cat([
        torch.arange(batch_size, device=z.device) + batch_size,
        torch.arange(batch_size, device=z.device)
    ])

    # Labels
    labels = positives

    # Loss
    loss = F.cross_entropy(sim_matrix, labels)
    return loss
def ntxent_loss_with_labels(out0, labels, temperature=0.5):
    """
    NT-Xent loss for a batch with labels indicating class/group similarity.

    Args:
        out0 (Tensor): Output embeddings for a batch of images (batch_size, embedding_size).
        labels (Tensor): Labels for the batch (batch_size,). Each label indicates the class/group of each sample.
        temperature (float): Scaling factor for logits.

    Returns:
        Tensor: Contrastive Cross-Entropy Loss value.
    """
    
    # Normalize the output to unit length (cosine similarity)
    out0 = nn.functional.normalize(out0, dim=1)

    # Calculate cosine similarity (pairwise similarity matrix)
    logits = torch.einsum("nc,mc->nm", out0, out0) / temperature
    print(logits)
    # Mask diagonal (self-similarity)
    batch_size = out0.size(0)
    mask = torch.eye(batch_size, device=out0.device, dtype=torch.bool)
    logits = logits[~mask].view(batch_size, -1)  # Remove self-similarities

    # Generate positive pair labels: Same label is a positive pair
    labels = labels.unsqueeze(0) == labels.unsqueeze(1)  # Shape: (batch_size, batch_size)
    print(labels)
    # Convert boolean mask to integer (1 for positive pairs, 0 for negative pairs)
    labels = labels.float()

    # Cross-entropy loss: maximize similarity for positive pairs, minimize for negative pairs
    cross_entropy = nn.CrossEntropyLoss(reduction="mean")
    print(labels)
    # The target labels should be indices where labels are 1 (positive pairs)
    target = labels.argmax(dim=1)  # Get the index of the positive pair for each sample
    print(target)
    logits = logits.to("cuda:0")
    target = target.to("cuda:0")
    # Calculate the loss
    loss = cross_entropy(logits, target)

    return loss
directory="/home/chandan/dmc/segment_audio"
batch_size=128
#dataset_files = [os.path.join(directory, subdir, file) for subdir in os.listdir(directory) for file in os.listdir(os.path.join(directory, subdir))]
dataset = ContrastiveAudioDataset(directory)
print(len(dataset.labels))
sampler = DistinctClassBatchSampler(dataset.labels, batch_size)
encodingDataLoader = DataLoader(dataset,batch_sampler=sampler)
def plot_spectrogram(spectrogram, title):
    spectrogram = spectrogram.squeeze().cpu().numpy() 
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='magma')
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("dummy.png")

# Set the device to CUDA if available, otherwise fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BirdSoundModel(input_channels=1, embed_size=128, num_heads=4, projection_dim=128).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
#print(len(dataset))

for s1,label in encodingDataLoader:
    print(s1.shape, label)
    s1 = s1.to(device) # (B, 1, H, W)
    #view2 = s2.unsqueeze(1).to(device)

    out1 = model(s1)
    #out2 = model(view2)

    loss = ntxent_loss_with_labels(out1, label, temperature=0.5)
    print(loss)
    """print(anchor.shape, positive.shape, negative.shape)
    anchor = anchor.unsqueeze(1).to(device)
    positive = positive.unsqueeze(1).to(device)
    negative = negative.unsqueeze(1).to(device)

    print(anchor.shape, positive.shape, negative.shape)
    output_anchor = model(anchor)
    output_positive = model(positive)
    output_negative = model(negative)

    anchor_avg = output_anchor.mean(dim=1)  # Shape becomes (batch_size, 128)
    positive_avg = output_positive.mean(dim=1)  # Shape becomes (batch_size, 128)
    negative_avg = output_negative.mean(dim=1)
    #print(anchor_avg.shape)
    #print(nt_xent_loss_three(anchor_avg, positive_avg, negative_avg))"""
    break

print(len(encodingDataLoader))
#fig, axs = plt.subplots(1, 3, figsize=(15, 5))

#for ax, spec, title in zip(axs, [anchor[0], positive[0], negative[0]], ["Anchor", "Positive", "Negative"]):
#    ax.imshow(spec.squeeze().cpu().numpy(), aspect='auto', origin='lower', cmap='magma')
#    ax.set_title(title)
#    ax.axis('off')

#plt.tight_layout()
#plt.savefig("anchor_positive_negative.png")
"""import torch
from model import BirdSoundModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BirdSoundModel(input_channels=1, embed_size=128, num_heads=4, num_experts=3, projection_dim=128).to(device)

checkpoint = torch.load('feature_extractor.pth', map_location=device)

model.cnn_block1.load_state_dict(checkpoint['cnn_block1'])
model.cnn_block2.load_state_dict(checkpoint['cnn_block2'])
model.cnn_block3.load_state_dict(checkpoint['cnn_block3'])
model.transformer_layer1.load_state_dict(checkpoint['transformer_layer1'])
model.transformer_layer2.load_state_dict(checkpoint['transformer_layer2'])
model.transformer_layer3.load_state_dict(checkpoint['transformer_layer3'])
model.moe.load_state_dict(checkpoint['moe'])

for param in model.proj_head.parameters():
    param.requires_grad = False

feature_extractor_modules = [
    model.cnn_block1,
    model.cnn_block2,
    model.cnn_block3,
    model.transformer_layer1,
    model.transformer_layer2,
    model.transformer_layer3,
    model.moe
]

total_params = sum(p.numel() for module in feature_extractor_modules for p in module.parameters() if p.requires_grad)

print(f"Total number of parameters in feature extractor: {total_params}")

"""