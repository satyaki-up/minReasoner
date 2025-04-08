from dataclasses import dataclass
from typing import Optional


@dataclass
class PPOTrainerConfig:
    """Configuration for PPO trainer."""
    
    learning_rate: float = 1e-5
    batch_size: int = 8
    max_length: int = 512
    device: str = "cpu"
    epsilon: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_epochs_per_update: int = 4
    num_epochs: int = 3
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    output_dir: Optional[str] = None


@dataclass
class DPOTrainerConfig:
    """Configuration for DPO trainer."""
    
    learning_rate: float = 1e-5
    batch_size: int = 8
    max_length: int = 512
    device: str = "cpu"
    beta: float = 0.1
    num_epochs: int = 3
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    output_dir: Optional[str] = None 