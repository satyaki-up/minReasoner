from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.optim import Adam
from transformers import PreTrainedModel, PreTrainedTokenizer

from .base import BaseTrainer


class DPOTrainer(BaseTrainer):
    """Direct Preference Optimization trainer for language models."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        learning_rate: float = 1e-5,
        batch_size: int = 8,
        max_length: int = 512,
        device: str = "cpu",
        beta: float = 0.1,
    ):
        """
        Initialize the DPO trainer.
        
        Args:
            model: The language model to fine-tune
            tokenizer: The tokenizer for the model
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
            max_length: Maximum sequence length
            device: Device to use for training ("cpu" or "cuda")
            beta: DPO beta parameter for KL constraint
        """
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )
        self.beta = beta
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        
    def compute_dpo_loss(
        self,
        chosen_logps: torch.Tensor,
        rejected_logps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the DPO loss.
        
        Args:
            chosen_logps: Log probabilities of chosen responses
            rejected_logps: Log probabilities of rejected responses
            
        Returns:
            DPO loss
        """
        losses = -F.logsigmoid(self.beta * (chosen_logps - rejected_logps))
        return losses.mean()
    
    def train(
        self,
        train_dataset: Any,
        num_epochs: int = 3,
        **kwargs
    ) -> Dict[str, float]:
        """
        Train the model using DPO.
        
        Args:
            train_dataset: The dataset to train on
            num_epochs: Number of training epochs
            **kwargs: Additional training arguments
            
        Returns:
            Dictionary containing training metrics
        """
        metrics = {
            "loss": [],
            "chosen_rewards": [],
            "rejected_rewards": [],
        }
        
        for epoch in range(num_epochs):
            epoch_metrics = self._train_epoch(train_dataset, **kwargs)
            for k, v in epoch_metrics.items():
                metrics[k].append(v)
                
        return metrics
    
    def _train_epoch(
        self,
        dataset: Any,
        **kwargs
    ) -> Dict[str, float]:
        """
        Train for one epoch using DPO.
        
        Args:
            dataset: The dataset to train on
            **kwargs: Additional training arguments
            
        Returns:
            Dictionary containing epoch metrics
        """
        self.model.train()
        total_loss = 0
        total_chosen_rewards = 0
        total_rejected_rewards = 0
        num_batches = 0
        
        # Create data loader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda x: x
        )
        
        for batch in dataloader:
            # Prepare batch data
            prompts = [item["prompt"] for item in batch]
            chosen_responses = [item["chosen"] for item in batch]
            rejected_responses = [item["rejected"] for item in batch]
            
            # Tokenize inputs
            input_ids = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).input_ids.to(self.device)
            
            chosen_ids = self.tokenizer(
                chosen_responses,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).input_ids.to(self.device)
            
            rejected_ids = self.tokenizer(
                rejected_responses,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).input_ids.to(self.device)
            
            # Get log probabilities for chosen and rejected responses
            chosen_outputs = self.model(
                input_ids=input_ids,
                labels=chosen_ids
            )
            rejected_outputs = self.model(
                input_ids=input_ids,
                labels=rejected_ids
            )
            
            chosen_logps = chosen_outputs.logits.log_softmax(dim=-1)
            rejected_logps = rejected_outputs.logits.log_softmax(dim=-1)
            
            # Compute DPO loss
            losses = self.compute_dpo_loss(chosen_logps, rejected_logps)
            
            # Optimize
            self.optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # Update metrics
            total_loss += losses.item()
            total_chosen_rewards += chosen_logps.mean().item()
            total_rejected_rewards += rejected_logps.mean().item()
            num_batches += 1
        
        # Compute average metrics
        avg_metrics = {
            "loss": total_loss / max(1, num_batches),
            "chosen_rewards": total_chosen_rewards / max(1, num_batches),
            "rejected_rewards": total_rejected_rewards / max(1, num_batches),
        }
        
        return avg_metrics
    
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
        self.model.eval()
        metrics = {
            "eval_loss": 0,
            "eval_chosen_rewards": 0,
            "eval_rejected_rewards": 0,
        }
        
        # TODO: Implement evaluation
        # This is a placeholder for the actual implementation
        
        return metrics 