import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from minReasoner import DPOTrainer

def create_synthetic_dataset():
    """Create a small synthetic preference dataset."""
    data = {
        "chosen": [
            "The solution to climate change requires a multi-faceted approach including renewable energy adoption, carbon capture, and policy changes.",
            "To solve this math problem, let's break it down into steps: first calculate the base case, then apply the formula.",
            "The evidence suggests that regular exercise and a balanced diet are key factors in maintaining good health.",
        ],
        "rejected": [
            "Climate change isn't real, it's just natural weather patterns.",
            "The answer is 42, trust me.",
            "Just take these pills and you'll be healthy, no need for lifestyle changes.",
        ],
    }
    return Dataset.from_dict(data)

def main():
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model and tokenizer
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    # Create synthetic dataset
    dataset = create_synthetic_dataset()
    print("Dataset created with size:", len(dataset))

    # Initialize DPO trainer
    trainer = DPOTrainer(
        model=model,
        tokenizer=tokenizer,
        learning_rate=1e-5,
        batch_size=2,
        max_length=128,
        device=device,
        beta=0.1,  # DPO hyperparameter
    )

    # Train the model
    print("Starting training...")
    metrics = trainer.train(
        train_dataset=dataset,
        num_epochs=3,
    )
    print("Training completed!")
    print("Final metrics:", metrics)

    # Save the fine-tuned model
    output_dir = "gpt2_dpo_finetuned"
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")

    # Test the model
    prompt = "The best way to solve climate change is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    print("\nTesting the model:")
    print("Prompt:", prompt)
    
    generated_ids = model.generate(
        input_ids,
        max_length=100,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print("Generated response:", response)

if __name__ == "__main__":
    main() 