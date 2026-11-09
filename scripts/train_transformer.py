import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformer_model import FallTransformer
import numpy as np

def train_transformer(epochs=5, batch_size=16):
    num_classes = 5
    batch_size = 16
    # 1. Model Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Training on {device}")
    
    model = FallTransformer(17, 128, 4, 2, 512, 5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    # Use GradScaler for AMP (if CUDA)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    # 2. Compile model for 30-50% speedup (SOTA)
    print("📦 Compiling Transformer with torch.compile...")
    try:
        model = torch.compile(model, mode='reduce-overhead')
        print("✅ Transformer compiled successfully")
    except Exception as e:
        print(f"⚠️ Compilation failed: {e}. Falling back.")
    try:
        X_real = np.load("X_sequences.npy")
        y_real = np.load("y_labels.npy")
        X_syn = np.load("X_synthetic.npy")
        y_syn = np.load("y_synthetic.npy")
        
        # Mix for demo with SOTA remapping
        y_real_remapped = np.copy(y_real)
        y_real_remapped[y_real == 1] = 2 # Falling (Old 1 -> New 2)
        y_real_remapped[y_real == 2] = 4 # Fallen (Old 2 -> New 4)
        X = np.concatenate([X_real, X_syn[:500]], axis=0)
        y = np.concatenate([y_real_remapped, y_syn[:500]], axis=0)
        
        X = torch.FloatTensor(X)
        y = torch.LongTensor(y)
        print(f"✅ Loaded {len(X)} samples for training.")
    except Exception as e:
        print(f"⚠️ Failed to load real data: {e}. Using synthetic/dummy.")
        X = torch.randn(100, 30, 51)
        y = torch.randint(0, 3, (100,))

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                        num_workers=4, pin_memory=(device.type == 'cuda'))

    # 3. Training Loop
    print(f"🏁 Starting Training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
            
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss/len(loader):.4f} | Acc: {100.*correct/total:.2f}%")

    # 4. Save Model
    torch.save(model.state_dict(), 'transformer_fall_detection.pth')
    print("✅ Model saved to transformer_fall_detection.pth")

if __name__ == "__main__":
    train_transformer(epochs=50)
