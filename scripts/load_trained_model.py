#!/usr/bin/env python3
"""
Load and use a trained model
Example script showing how to load the trained model for inference
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from training.models import get_model


def load_model(checkpoint_path: str, model_type: str = 'simple_cnn', num_classes: int = 10):
    """
    Load a trained model from checkpoint

    Args:
        checkpoint_path: Path to the checkpoint file (.pth)
        model_type: Type of model architecture
        num_classes: Number of output classes

    Returns:
        model: Loaded PyTorch model ready for inference
    """
    print(f"Loading model from: {checkpoint_path}")

    # Create model architecture
    model = get_model(model_type, num_classes=num_classes, pretrained=False)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Check if it's a full checkpoint or just weights
    if 'model_state_dict' in checkpoint:
        # Full checkpoint with optimizer, epoch, etc.
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"✓ Loaded checkpoint from epoch: {epoch}")
    else:
        # Just model weights
        model.load_state_dict(checkpoint)
        print(f"✓ Loaded model weights")

    # Set to evaluation mode
    model.eval()

    print(f"✓ Model ready for inference")
    return model


def run_inference(model, input_tensor):
    """
    Run inference on input data

    Args:
        model: Loaded PyTorch model
        input_tensor: Input data (batch_size, channels, height, width)

    Returns:
        predictions: Model predictions
    """
    with torch.no_grad():
        output = model(input_tensor)
        predictions = torch.argmax(output, dim=1)

    return predictions, output


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Load and test trained model")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--model', type=str, default='simple_cnn',
                       help='Model architecture')
    parser.add_argument('--num-classes', type=int, default=10,
                       help='Number of classes')
    parser.add_argument('--test', action='store_true',
                       help='Run a test inference')

    args = parser.parse_args()

    # Load model
    model = load_model(args.checkpoint, args.model, args.num_classes)

    # Test inference if requested
    if args.test:
        print("\nRunning test inference...")

        # Create dummy input (batch_size=1, channels=3, height=32, width=32)
        test_input = torch.randn(1, 3, 32, 32)

        # Run inference
        predictions, logits = run_inference(model, test_input)

        print(f"✓ Test completed successfully")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {logits.shape}")
        print(f"  Predicted class: {predictions.item()}")
        print(f"  Logits: {logits.squeeze().tolist()[:5]}...")  # Show first 5

    print("\n" + "="*60)
    print("Model loaded successfully!")
    print("="*60)
    print("\nExample usage:")
    print(f"""
import torch
from load_trained_model import load_model, run_inference

# Load model
model = load_model('{args.checkpoint}', '{args.model}', {args.num_classes})

# Prepare your data
input_data = torch.randn(1, 3, 32, 32)  # Your actual data here

# Run inference
predictions, logits = run_inference(model, input_data)
print(f"Predicted class: {{predictions.item()}}")
""")


if __name__ == '__main__':
    main()
