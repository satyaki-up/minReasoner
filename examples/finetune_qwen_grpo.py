import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from minReasoner import GRPOTrainer

def create_synthetic_dataset():
    """Create a small synthetic dataset with groups of responses."""
    data = {
        "prompt": [
            "What is the best way to solve climate change?",
            "How can we improve education?",
            "What are the benefits of exercise?",
        ],
        "responses": [
            [
                "The solution to climate change requires a multi-faceted approach including renewable energy adoption, carbon capture, and policy changes.",
                "We should focus on planting more trees to absorb carbon dioxide.",
                "Individuals should reduce their carbon footprint by using public transportation.",
                "Governments should implement stricter regulations on emissions.",
            ],
            [
                "To improve education, we need better funding, qualified teachers, and modern curriculum.",
                "Online learning platforms can make education more accessible.",
                "Parental involvement is crucial for student success.",
                "Standardized testing should be reformed to focus on critical thinking.",
            ],
            [
                "Regular exercise improves cardiovascular health, mental well-being, and longevity.",
                "Exercise helps maintain a healthy weight and build muscle strength.",
                "Physical activity can reduce stress and improve sleep quality.",
                "Exercise is important for maintaining bone density as we age.",
            ],
        ],
        "rewards": [
            [0.9, 0.6, 0.5, 0.7],
            [0.8, 0.7, 0.6, 0.5],
            [0.9, 0.8, 0.7, 0.6],
        ],
    }
    return Dataset.from_dict(data)

def main():
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model and tokenizer
    model_name = "Qwen/Qwen2.5-Math-1.5B"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    # Create synthetic dataset
    dataset = create_synthetic_dataset()
    print("Dataset created with size:", len(dataset))

    # Initialize GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        learning_rate=1e-5,
        batch_size=1,  # Use batch size of 1 to avoid issues
        max_length=128,
        device=device,
        beta=0.1,  # GRPO hyperparameter
        group_size=4,  # Number of responses per group
        temperature=0.1,  # Temperature for softmax
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
    output_dir = "gpt2_grpo_finetuned"
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")

    # Test the model
    prompt = "What is AI?"
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