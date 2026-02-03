"""Export trained model weights for RL compatibility."""

import argparse
import os

import torch

from .model import MLPPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export trained model for older PyTorch compatibility"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="training/supervised/checkpoints/best_model.pt",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: same dir as checkpoint, with _exported suffix)",
    )

    return parser.parse_args()


def export_model(checkpoint_path: str, output_path: str = None) -> str:
    """Export model weights for older PyTorch compatibility.

    Args:
        checkpoint_path: Path to the training checkpoint.
        output_path: Output path for exported model. If None, uses checkpoint
            directory with _exported suffix.

    Returns:
        Path to the exported model file.
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extract config and state dict
    config = checkpoint["config"]
    state_dict = checkpoint["model_state_dict"]

    # Reconstruct model to verify state dict
    model = MLPPolicy(
        obs_dim=config["obs_dim"],
        action_dim=config["action_dim"],
        hidden_sizes=config["hidden_sizes"],
    )
    model.load_state_dict(state_dict)

    # Prepare metadata
    metadata = {
        "obs_dim": config["obs_dim"],
        "action_dim": config["action_dim"],
        "hidden_sizes": config["hidden_sizes"],
        "training_info": {
            "epoch": checkpoint.get("epoch"),
            "train_loss": checkpoint.get("train_loss"),
            "val_loss": checkpoint.get("val_loss"),
        },
    }

    # Determine output path
    if output_path is None:
        base, ext = os.path.splitext(checkpoint_path)
        output_path = f"{base}_exported{ext}"

    # Save with old zipfile serialization for compatibility
    export_data = {
        "state_dict": state_dict,
        "metadata": metadata,
    }

    torch.save(
        export_data,
        output_path,
        _use_new_zipfile_serialization=False,
    )

    print(f"Exported model to: {output_path}")
    print(f"Metadata: {metadata}")

    return output_path


def load_exported_model(export_path: str) -> MLPPolicy:
    """Load an exported model.

    Args:
        export_path: Path to the exported model file.

    Returns:
        Loaded MLPPolicy model in eval mode.
    """
    export_data = torch.load(export_path, map_location="cpu")

    metadata = export_data["metadata"]
    state_dict = export_data["state_dict"]

    model = MLPPolicy(
        obs_dim=metadata["obs_dim"],
        action_dim=metadata["action_dim"],
        hidden_sizes=metadata["hidden_sizes"],
    )
    model.load_state_dict(state_dict)
    model.eval()

    return model


def main():
    args = parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return

    export_model(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
