"""
Model Definitions for Benchmarking
ResNet-50, BERT, GPT-2
"""

import torch
import torch.nn as nn
from torchvision import models
from transformers import BertModel, BertConfig, GPT2Model, GPT2Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_resnet50(num_classes: int = 1000, pretrained: bool = False) -> nn.Module:
    """
    Get ResNet-50 model

    Args:
        num_classes: Number of output classes
        pretrained: Load pretrained weights

    Returns:
        ResNet-50 model
    """
    model = models.resnet50(pretrained=pretrained)

    # Modify final layer if needed
    if num_classes != 1000:
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    logger.info(f"Created ResNet-50 (classes={num_classes}, pretrained={pretrained})")
    return model


def get_bert_base(num_classes: int = 2, pretrained: bool = False) -> nn.Module:
    """
    Get BERT-base model for classification

    Args:
        num_classes: Number of output classes
        pretrained: Load pretrained weights

    Returns:
        BERT model with classification head
    """
    if pretrained:
        model = BertModel.from_pretrained('bert-base-uncased')
    else:
        config = BertConfig(
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=512,
        )
        model = BertModel(config)

    # Add classification head
    class BertClassifier(nn.Module):
        def __init__(self, bert_model, num_classes):
            super().__init__()
            self.bert = bert_model
            self.classifier = nn.Linear(768, num_classes)

        def forward(self, input_ids, attention_mask=None):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
            logits = self.classifier(pooled_output)
            return logits

    classifier = BertClassifier(model, num_classes)
    logger.info(f"Created BERT-base (classes={num_classes}, pretrained={pretrained})")
    return classifier


def get_gpt2_small(num_classes: int = 2, pretrained: bool = False) -> nn.Module:
    """
    Get GPT-2 small model for classification

    Args:
        num_classes: Number of output classes
        pretrained: Load pretrained weights

    Returns:
        GPT-2 model with classification head
    """
    if pretrained:
        model = GPT2Model.from_pretrained('gpt2')
    else:
        config = GPT2Config(
            vocab_size=50257,
            n_positions=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
        )
        model = GPT2Model(config)

    # Add classification head
    class GPT2Classifier(nn.Module):
        def __init__(self, gpt2_model, num_classes):
            super().__init__()
            self.gpt2 = gpt2_model
            self.classifier = nn.Linear(768, num_classes)

        def forward(self, input_ids, attention_mask=None):
            outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
            # Use last token's hidden state
            last_hidden_state = outputs.last_hidden_state[:, -1, :]
            logits = self.classifier(last_hidden_state)
            return logits

    classifier = GPT2Classifier(model, num_classes)
    logger.info(f"Created GPT-2 small (classes={num_classes}, pretrained={pretrained})")
    return classifier


def get_simple_cnn(num_classes: int = 10) -> nn.Module:
    """
    Get simple CNN for quick testing

    Args:
        num_classes: Number of output classes

    Returns:
        Simple CNN model
    """
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(256 * 4 * 4, 512)
            self.fc2 = nn.Linear(512, num_classes)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = self.pool(self.relu(self.conv3(x)))
            x = x.view(x.size(0), -1)
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    model = SimpleCNN(num_classes)
    logger.info(f"Created SimpleCNN (classes={num_classes})")
    return model


def get_model(model_name: str, num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """
    Factory function to get model by name

    Args:
        model_name: Model name ('resnet50', 'bert', 'gpt2', 'simple_cnn')
        num_classes: Number of output classes
        pretrained: Load pretrained weights

    Returns:
        Model
    """
    model_name = model_name.lower()

    if model_name == 'resnet50':
        return get_resnet50(num_classes, pretrained)
    elif model_name == 'bert':
        return get_bert_base(num_classes, pretrained)
    elif model_name == 'gpt2':
        return get_gpt2_small(num_classes, pretrained)
    elif model_name == 'simple_cnn':
        return get_simple_cnn(num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model: nn.Module, model_name: str = "Model"):
    """Print model information"""
    total_params = count_parameters(model)
    print(f"\n{model_name} Info:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Size: {total_params * 4 / (1024**2):.2f} MB (float32)")


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")

    models_to_test = [
        ('simple_cnn', 10),
        ('resnet50', 1000),
    ]

    for model_name, num_classes in models_to_test:
        try:
            model = get_model(model_name, num_classes, pretrained=False)
            print_model_info(model, model_name)
        except Exception as e:
            logger.error(f"Failed to create {model_name}: {e}")
