import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import glob

from model import DRResNet


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focal Loss down-weights easy examples and focuses training on hard negatives.
    
    Args:
        alpha (float): Weighting factor for balancing positive/negative examples
        gamma (float): Focusing parameter (higher = more focus on hard examples)
        reduction (str): 'mean', 'sum', or 'none'
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Compute focal loss.
        
        Args:
            inputs: Model outputs (logits), shape [B, C]
            targets: Ground truth labels, shape [B]
            
        Returns:
            Focal loss scalar
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        probs = F.softmax(inputs, dim=1)
        target_probs = probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        focal_factor = (1 - target_probs) ** self.gamma
        loss = self.alpha * focal_factor * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def load_data_from_csv(data_dir):
    """
    Load and process all CSV files from the data directory.
    
    Each CSV file contains columns: Time (s), HB (mM), Glu (mM), pH, label
    Features: HB, Glu, pH
    Target: label (0-3 representing different metabolic states)
    
    Args:
        data_dir (str): Path to directory containing CSV files
        
    Returns:
        tuple: (features, labels) as numpy arrays
    """
    all_features = []
    all_labels = []
    
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        
        # Extract features: HB, Glu, pH
        features = df[['HB (mM)', 'Glu (mM)', 'pH']].values
        labels = df['label'].values
        
        all_features.append(features)
        all_labels.append(labels)
    
    # Concatenate all data
    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)
    
    return features, labels


def train_model(model, train_loader, val_loader, criterion, optimizer, 
                device, epochs=500):
    """
    Train the model.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use (cuda/cpu)
        epochs: Number of training epochs
        
    Returns:
        dict: Training history containing losses and accuracies
    """
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, "
                  f"Val Acc: {val_acc:.2f}%")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history


def main():
    """
    Main function to run training.
    """
    # Configuration
    DATA_DIR = "data"
    BATCH_SIZE = 32
    EPOCHS = 500
    LEARNING_RATE = 0.0001
    HIDDEN_DIM = 16
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    features, labels = load_data_from_csv(DATA_DIR)
    print(f"Dataset shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Unique labels: {np.unique(labels)}")
    
    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    print(f"Train set: {X_train.shape}, Val set: {X_val.shape}")
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    
    # Data loaders
    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_tensor, y_val_tensor),
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    # Model
    input_dim = X_train.shape[1]
    model = DRResNet(input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_classes=4)
    # Trasfer Learning
    # model.load_state_dict(torch.load('drresnet_model.pth'), strict=False)
    
    model = model.to(device)
    print(f"Model: DRResNet")
    print(f"Input dim: {input_dim}, Hidden dim: {HIDDEN_DIM}")
    
    # Loss and optimizer
    criterion = FocalLoss(alpha=1, gamma=2)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # Train
    print("\nStarting training...")
    history = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        device, epochs=EPOCHS
    )
    
    # Save model
    torch.save(model.state_dict(), "drresnet_model.pth")
    print("\nModel saved to drresnet_model.pth")
    
    # Final evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    final_acc = 100 * correct / total
    print(f"\nFinal Validation Accuracy: {final_acc:.2f}%")
    print(f"Best Validation Accuracy: {max(history['val_acc']):.2f}%")


if __name__ == "__main__":
    main()
