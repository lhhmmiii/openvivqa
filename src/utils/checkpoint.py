import torch
from pathlib import Path


def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    train_loss,
    val_loss,
):
    """Save a training checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }

    torch.save(checkpoint, path)


def load_checkpoint(path, model, device, optimizer=None):
    """Load a training checkpoint.

    Returns:
        checkpoint dict with keys: epoch, train_loss, val_loss, etc.
    """
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
