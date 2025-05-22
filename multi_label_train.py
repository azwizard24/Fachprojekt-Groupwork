import torch
import torch.nn.functional as F
from torchmetrics.classification import MultilabelAUROC
from torch.utils.data import DataLoader
from birdset_model import ConvNextBirdSet
from multi_label_dataset import MultiLabelAudioDataset, load_dataloaders
import os

class BCEFocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: raw logits
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        p_t = targets * probs + (1 - targets) * (1 - probs)  # pt for each element
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        modulating_factor = (1 - p_t) ** self.gamma

        loss = alpha_factor * modulating_factor * BCE_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # no reduction

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
        probs = torch.sigmoid(y_hat)
        roc_auc_metric.update(probs, y.int())

        if batch_idx % 2 == 0:
            print(f"[Train] Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    macro_auc = roc_auc_metric.compute().mean()
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

            probs = torch.sigmoid(y_hat)
            roc_auc_metric.update(probs, y.int())

    avg_loss = total_loss / len(dataloader)
    macro_auc = roc_auc_metric.compute().mean()
    print(f"[Test] Epoch {epoch} Completed - Avg Loss: {avg_loss:.4f}, Macro AUC: {macro_auc:.4f}")
    return avg_loss, macro_auc

def save_model(model, epoch, save_dir="checkpoints_multi_label"):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, f"birdset_epoch{epoch}.pt"))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    annotation_file = "multi_label_predictions.json"
    train_loader, test_loader, num_classes = load_dataloaders(annotation_file)

    model = BirdsetModule(num_classes=num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = BCEFocalLoss(reduction='mean')

    train_roc_auc = MultilabelAUROC(num_labels=num_classes, average=None).to(device)
    test_roc_auc = MultilabelAUROC(num_labels=num_classes, average=None).to(device)

    for epoch in range(1, 6):
        train_one_epoch(model, train_loader, optimizer, device, criterion, train_roc_auc, epoch)
        evaluate(model, test_loader, device, criterion, test_roc_auc, epoch)
        save_model(model, epoch)

if __name__ == "__main__":
    main()
