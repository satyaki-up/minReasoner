from typing import Dict, List, Optional, Union

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer


def prepare_dataset(
    dataset: Union[Dataset, str],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    batch_size: int = 8,
) -> torch.utils.data.DataLoader:
    """
    Prepare a dataset for training.
    
    Args:
        dataset: HuggingFace dataset or dataset name
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        batch_size: Batch size for training
        
    Returns:
        DataLoader for the prepared dataset
    """
    if isinstance(dataset, str):
        dataset = Dataset.load_from_disk(dataset)
        
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    return torch.utils.data.DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=True,
    )


def prepare_preference_dataset(
    dataset: Union[Dataset, str],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    batch_size: int = 8,
) -> torch.utils.data.DataLoader:
    """
    Prepare a preference dataset for DPO training.
    
    Args:
        dataset: HuggingFace dataset or dataset name
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        batch_size: Batch size for training
        
    Returns:
        DataLoader for the prepared dataset
    """
    if isinstance(dataset, str):
        dataset = Dataset.load_from_disk(dataset)
        
    def tokenize_function(examples):
        chosen = tokenizer(
            examples["chosen"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        rejected = tokenizer(
            examples["rejected"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
        }
        
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    return torch.utils.data.DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=True,
    ) 