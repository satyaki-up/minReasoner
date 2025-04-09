# minReasoner Examples

This directory contains example scripts demonstrating how to use the minReasoner library for fine-tuning language models.

## GPT-2 DPO Fine-tuning Example

The `finetune_gpt2_dpo.py` script demonstrates how to fine-tune GPT-2 using Direct Preference Optimization (DPO). This example:

1. Creates a small synthetic dataset with preferred and rejected responses
2. Loads GPT-2 model and tokenizer
3. Fine-tunes the model using DPO
4. Tests the fine-tuned model with a sample prompt

### Running the Example

1. Make sure you have minReasoner installed:
```bash
pip install -e .
```

2. Install additional dependencies:
```bash
pip install torch transformers datasets
```

3. Run the example:
```bash
python finetune_gpt2_dpo.py
```

The script will:
- Train the model for 3 epochs
- Save the fine-tuned model to `gpt2_dpo_finetuned` directory
- Generate a sample response using the fine-tuned model

### Expected Output

The script will print:
- Training progress and metrics
- Final evaluation metrics
- A generated response to the test prompt

### Customization

You can modify:
- The synthetic dataset in `create_synthetic_dataset()`
- Training hyperparameters in the `DPOTrainer` initialization
- The test prompt
- Model generation parameters 