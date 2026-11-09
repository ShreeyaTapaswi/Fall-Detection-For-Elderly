from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from standing_dataset import CLASS_NAMES
from transformer_model_standing import StandingFallTransformer


def load_dataset(dataset_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    X_path = dataset_dir / "X_sequences.npy"
    y_path = dataset_dir / "y_labels.npy"
    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing extracted standing data in {dataset_dir}")

    X = torch.from_numpy(np.load(X_path)).float()
    y = torch.from_numpy(np.load(y_path)).long()
    return X, y


def train_transformer_standing(
    epochs: int = 30,
    batch_size: int = 16,
    dataset_dir: str = "artifacts/standing_transformer",
    output_name: str = "transformer_standing_fall_detection.pth",
):
    project_root = Path(__file__).resolve().parent.parent
    dataset_path = project_root / dataset_dir
    X, y = load_dataset(dataset_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StandingFallTransformer(num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    dataset = TensorDataset(X, y)
    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    best_state = None
    best_val_acc = 0.0

    print(f"Training standing-fall transformer on {device} with classes {CLASS_NAMES}")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            train_total += batch_y.size(0)
            train_correct += (predictions == batch_y).sum().item()

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_X)
                predictions = outputs.argmax(dim=1)
                val_total += batch_y.size(0)
                val_correct += (predictions == batch_y).sum().item()

        train_acc = 100.0 * train_correct / max(train_total, 1)
        val_acc = 100.0 * val_correct / max(val_total, 1)
        print(
            f"Epoch [{epoch + 1}/{epochs}] | "
            f"Loss: {running_loss / max(len(train_loader), 1):.4f} | "
            f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    output_path = project_root / output_name
    torch.save(best_state or model.state_dict(), output_path)
    print(f"Saved standing-fall transformer to {output_path}")


if __name__ == "__main__":
    train_transformer_standing()
