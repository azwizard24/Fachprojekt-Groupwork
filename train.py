from dataset import ContrastiveDataset
from model import BirdSoundModel
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader

# NT-Xent loss function
def nt_xent_loss(z1, z2, temperature=.5):
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

# ---------------- Setup ---------------- #

directory = "loop_test"
dataset_files = [os.path.join(directory, subdir, file) for subdir in os.listdir(directory) for file in os.listdir(os.path.join(directory, subdir))]
dataset = ContrastiveDataset(dataset_files)

batch_size = 8
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BirdSoundModel(input_channels=1, embed_size=128, num_heads=4, projection_dim=128).to(device)

optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=1e-3, 
    betas=(0.9, 0.999) 
)
num_epochs = 1
loss_history = []

# ---------------- Training ---------------- #

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for step, (view1, view2, _) in enumerate(dataloader):
        view1 = view1.unsqueeze(1).to(device)  # (B, 1, H, W)
        view2 = view2.unsqueeze(1).to(device)

        out1 = model(view1)
        out2 = model(view2)
        loss = nt_xent_loss(out1, out2, temperature=0.5)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (step + 1) % 2 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    loss_history.append(avg_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")
# ---------------- Saving ---------------- #

torch.save(model.state_dict(), "feature_extractor.pth")
# plt.figure(figsize=(8, 5))
# plt.plot(range(1, num_epochs + 1), loss_history, marker='o')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss Over Epochs')
# plt.grid(True)
# plt.savefig('training_loss_plot.png')
