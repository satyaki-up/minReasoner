from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


class BaseTrainer(ABC):
    """Base class for all RL trainers."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        learning_rate: float = 1e-5,
        batch_size: int = 8,
        max_length: int = 512,
        device: str = "cpu",
    ):
        """
        Initialize the base trainer.
        
        Args:
            model: The language model to fine-tune
            tokenizer: The tokenizer for the model
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
            max_length: Maximum sequence length
            device: Device to use for training ("cpu" or "cuda")
        """
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        
        # Move model to device
        self.model.to(self.device)
        
    @abstractmethod
    def train(
        self,
        train_dataset: Any,
        num_epochs: int = 3,
        **kwargs
    ) -> Dict[str, float]:
        """
        Train the model using the specified RL algorithm.
        
        Args:
            train_dataset: The dataset to train on
            num_epochs: Number of training epochs
            **kwargs: Additional training arguments
            
        Returns:
            Dictionary containing training metrics
        """
        pass
    
    @abstractmethod
    def evaluate(
        self,
        eval_dataset: Any,
        **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate the model on the given dataset.
        
        Args:
            eval_dataset: The dataset to evaluate on
            **kwargs: Additional evaluation arguments
            
        Returns:
            Dictionary containing evaluation metrics
        """
        pass
    
    def save_model(self, path: str) -> None:
        """
        Save the model and tokenizer.
        
        Args:
            path: Path to save the model to
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    def load_model(self, path: str) -> None:
        """
        Load a saved model and tokenizer.
        
        Args:
            path: Path to load the model from
        """
        self.model = self.model.__class__.from_pretrained(path)
        self.tokenizer = self.tokenizer.__class__.from_pretrained(path)
        self.model.to(self.device) 