# minReasoner

A minimal library for fine-tuning language models using reinforcement learning algorithms. This library implements various RL algorithms like PPO (Proximal Policy Optimization) and DPO (Direct Preference Optimization) to improve the reasoning abilities of language models.

## Features

- Support for HuggingFace models and datasets
- Implementation of RL algorithms (PPO, DPO)
- Local training support (no CUDA required)
- Simple and clean API for fine-tuning

## Installation

```bash
pip install -e .
```

## Usage

```python
from minReasoner import PPOTrainer, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load a model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Initialize a trainer
trainer = PPOTrainer(
    model=model,
    tokenizer=tokenizer,
    learning_rate=1e-5
)

# Train the model
trainer.train(
    train_dataset="your_dataset",
    num_epochs=3
)
```

## Project Structure

```
minReasoner/
├── minReasoner/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── ppo.py
│   │   └── dpo.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── data.py
│   └── config/
│       ├── __init__.py
│       └── default.py
├── setup.py
└── README.md
```

## License

MIT License 