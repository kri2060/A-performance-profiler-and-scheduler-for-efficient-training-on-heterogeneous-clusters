"""
Dataset utilities for training
Includes synthetic datasets and real dataset loaders
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyntheticImageDataset(Dataset):
    """Synthetic image dataset for quick testing"""

    def __init__(
        self,
        num_samples: int = 10000,
        image_size: int = 224,
        num_classes: int = 1000,
        channels: int = 3
    ):
        """
        Args:
            num_samples: Number of samples in dataset
            image_size: Image size (height and width)
            num_classes: Number of classes
            channels: Number of channels
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        self.channels = channels

        logger.info(
            f"Created SyntheticImageDataset: "
            f"{num_samples} samples, {image_size}x{image_size}, {num_classes} classes"
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random image
        image = torch.randn(self.channels, self.image_size, self.image_size)

        # Generate random label
        label = torch.randint(0, self.num_classes, (1,)).item()

        return image, label


class SyntheticTextDataset(Dataset):
    """Synthetic text dataset for NLP models"""

    def __init__(
        self,
        num_samples: int = 10000,
        sequence_length: int = 128,
        vocab_size: int = 30522,
        num_classes: int = 2
    ):
        """
        Args:
            num_samples: Number of samples
            sequence_length: Token sequence length
            vocab_size: Vocabulary size
            num_classes: Number of classes
        """
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.num_classes = num_classes

        logger.info(
            f"Created SyntheticTextDataset: "
            f"{num_samples} samples, length={sequence_length}"
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random token IDs
        input_ids = torch.randint(0, self.vocab_size, (self.sequence_length,))

        # Generate attention mask (all ones)
        attention_mask = torch.ones(self.sequence_length, dtype=torch.long)

        # Generate random label
        label = torch.randint(0, self.num_classes, (1,)).item()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
        }


def get_cifar10_loaders(
    data_dir: str = "./data",
    batch_size: int = 32,
    num_workers: int = 4,
    download: bool = True
):
    """
    Get CIFAR-10 data loaders

    Args:
        data_dir: Data directory
        batch_size: Batch size
        num_workers: Number of workers
        download: Download dataset if not present

    Returns:
        train_loader, test_loader
    """
    # Transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Datasets
    train_dataset = CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=transform_train
    )

    test_dataset = CIFAR10(
        root=data_dir,
        train=False,
        download=download,
        transform=transform_test
    )

    # Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    logger.info(f"Loaded CIFAR-10: {len(train_dataset)} train, {len(test_dataset)} test")

    return train_loader, test_loader


def get_cifar100_loaders(
    data_dir: str = "./data",
    batch_size: int = 32,
    num_workers: int = 4,
    download: bool = True
):
    """Get CIFAR-100 data loaders"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_dataset = CIFAR100(
        root=data_dir,
        train=True,
        download=download,
        transform=transform_train
    )

    test_dataset = CIFAR100(
        root=data_dir,
        train=False,
        download=download,
        transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    logger.info(f"Loaded CIFAR-100: {len(train_dataset)} train, {len(test_dataset)} test")

    return train_loader, test_loader


def get_dataset(
    dataset_name: str,
    num_samples: int = 10000,
    image_size: int = 224,
    sequence_length: int = 128,
    num_classes: int = 10,
    data_dir: str = "./data"
) -> Dataset:
    """
    Factory function to get dataset by name

    Args:
        dataset_name: Dataset name
        num_samples: Number of samples (for synthetic)
        image_size: Image size (for images)
        sequence_length: Sequence length (for text)
        num_classes: Number of classes
        data_dir: Data directory

    Returns:
        Dataset
    """
    dataset_name = dataset_name.lower()

    if dataset_name == 'synthetic_image':
        return SyntheticImageDataset(
            num_samples=num_samples,
            image_size=image_size,
            num_classes=num_classes
        )
    elif dataset_name == 'synthetic_text':
        return SyntheticTextDataset(
            num_samples=num_samples,
            sequence_length=sequence_length,
            num_classes=num_classes
        )
    elif dataset_name == 'cifar10':
        return CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        )
    elif dataset_name == 'cifar100':
        return CIFAR100(
            root=data_dir,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


if __name__ == "__main__":
    # Test datasets
    print("Testing datasets...")

    # Synthetic image
    dataset = SyntheticImageDataset(num_samples=100, image_size=224, num_classes=10)
    print(f"Synthetic image dataset: {len(dataset)} samples")
    image, label = dataset[0]
    print(f"  Image shape: {image.shape}, Label: {label}")

    # Synthetic text
    dataset = SyntheticTextDataset(num_samples=100, sequence_length=128)
    print(f"Synthetic text dataset: {len(dataset)} samples")
    sample = dataset[0]
    print(f"  Input IDs shape: {sample['input_ids'].shape}")
    print(f"  Attention mask shape: {sample['attention_mask'].shape}")
    print(f"  Label: {sample['label']}")
