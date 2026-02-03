"""Training script for behavioral cloning."""

import argparse
import os
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Adam

from .dataset import create_dataloaders
from .model import MLPPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train behavioral cloning policy")

    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--hidden-sizes",
        type=str,
        default="64,64",
        help="Hidden layer sizes, comma-separated",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.2, help="Validation split fraction"
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="training/expert_data",
        help="Directory containing expert data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="training/supervised/checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (epochs without improvement)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="roar-bc",
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Wandb run name (optional)",
    )

    return parser.parse_args()


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer: Adam,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for obs, actions in train_loader:
        obs = obs.to(device)
        actions = actions.to(device)

        optimizer.zero_grad()
        pred_actions = model(obs)
        loss = criterion(pred_actions, actions)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate(
    model: nn.Module,
    val_loader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Validate the model.

    Returns:
        Average validation loss.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for obs, actions in val_loader:
            obs = obs.to(device)
            actions = actions.to(device)

            pred_actions = model(obs)
            loss = criterion(pred_actions, actions)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def main():
    args = parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)

    # Parse hidden sizes
    hidden_sizes = [int(x) for x in args.hidden_sizes.split(",")]

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize wandb if enabled
    wandb_run: Optional[object] = None
    if args.wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config={
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "hidden_sizes": hidden_sizes,
                    "val_split": args.val_split,
                    "seed": args.seed,
                    "patience": args.patience,
                },
            )
        except ImportError:
            print("wandb not installed, disabling wandb logging")
            args.wandb = False

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
    )

    # Get data dimensions from first batch
    sample_obs, sample_actions = next(iter(train_loader))
    obs_dim = sample_obs.shape[1]
    action_dim = sample_actions.shape[1]
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")

    # Create model
    model = MLPPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
    ).to(device)
    print(f"Model: {model}")

    # Setup training
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Training loop with early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    print("\nStarting training...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        # Log metrics
        if args.wandb and wandb_run is not None:
            import wandb

            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
            )

        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            checkpoint_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "config": model.get_config(),
                },
                checkpoint_path,
            )
            print(f"  -> Saved best model (val_loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.1f}s")
    print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
    print(f"Best model saved to: {os.path.join(args.output_dir, 'best_model.pt')}")

    # Final checkpoint
    final_checkpoint_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": model.get_config(),
        },
        final_checkpoint_path,
    )
    print(f"Final model saved to: {final_checkpoint_path}")

    if args.wandb and wandb_run is not None:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
