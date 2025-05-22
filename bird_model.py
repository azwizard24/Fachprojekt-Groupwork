import torch
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAUROC
from torch.utils.data import DataLoader
from birdset_model import ConvNextBirdSet
#from bird_dataset import load_dataloaders
import os
from focal_loss_file import FocalLoss

class BirdsetModule(torch.nn.Module):
    def __init__(self, num_classes=206):
        super().__init__()
        self.model = ConvNextBirdSet(num_classes=num_classes)

    def forward(self, x):
        preprocessed = self.model.preprocess(x)
        return self.model(preprocessed)

def train_one_epoch(model, dataloader, optimizer, device, criterion, roc_auc_metric, epoch):
    model.train()
    total_loss = 0.0
    roc_auc_metric.reset()

    for batch_idx, batch in enumerate(dataloader):
        x = batch["audio"].to(device)
        y = batch["label"].to(device)

        optimizer.zero_grad()
        y_hat = model(x)

        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        probs = F.softmax(y_hat, dim=1)
        roc_auc_metric.update(probs, y)

        if batch_idx % 2 == 0:
            print(f"[Train] Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    aucs = roc_auc_metric.compute()
    valid_mask = aucs != 0
    macro_auc = aucs[valid_mask].mean() if valid_mask.any() else torch.tensor(0.0)
    print(f"[Train] Epoch {epoch} Completed - Avg Loss: {avg_loss:.4f}, Macro AUC: {macro_auc:.4f}")
    return avg_loss, macro_auc

def evaluate(model, dataloader, device, criterion, roc_auc_metric, epoch):
    model.eval()
    total_loss = 0.0
    roc_auc_metric.reset()

    with torch.no_grad():
        for batch in dataloader:
            x = batch["audio"].to(device)
            y = batch["label"].to(device)

            y_hat = model(x)
            loss = criterion(y_hat, y)
            total_loss += loss.item()

            probs = F.softmax(y_hat, dim=1)
            roc_auc_metric.update(probs, y)

    avg_loss = total_loss / len(dataloader)
    aucs = roc_auc_metric.compute()
    valid_mask = aucs != 0
    macro_auc = aucs[valid_mask].mean() if valid_mask.any() else torch.tensor(0.0)
    print(f"[Test] Epoch {epoch} Completed - Avg Loss: {avg_loss:.4f}, Macro AUC: {macro_auc:.4f}")
    return avg_loss, macro_auc

def save_model(model, epoch, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, f"birdset_epoch{epoch}.pt"))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BirdsetModule().to(device)

    root_dir = "segment_audio"
    train_dataloader, test_dataloader = load_dataloaders(root_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    criterion = FocalLoss(gamma=2.0, alpha=1.0)
    train_roc_auc = MulticlassAUROC(num_classes=206, average=None).to(device)
    test_roc_auc = MulticlassAUROC(num_classes=206, average=None).to(device)

    num_epochs = 5

    for epoch in range(1, num_epochs + 1):
        train_loss, train_auc = train_one_epoch(model, train_dataloader, optimizer, device, criterion, train_roc_auc, epoch)
        test_loss, test_auc = evaluate(model, test_dataloader, device, criterion, test_roc_auc, epoch)
        save_model(model, epoch)

if __name__ == "__main__":
    main()
