from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch

# Load the MentalChat16K dataset
print("Loading dataset...")
dataset = load_dataset("ShenLab/MentalChat16K")
train_data = dataset["train"]

# Prepare the tokenizer and model
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token for GPT-2

# Filter out examples with missing fields
def has_all_fields(example):
    return example["instruction"] and example["input"] and example["output"]

filtered_data = train_data.filter(has_all_fields)

# Filter for relevance using keywords
keywords = ["anxiety", "depression", "stress", "help", "support", "sad", "lonely", "cope", "mental health"]
def is_relevant(example):
    return any(kw in example["input"].lower() for kw in keywords)

relevant_data = filtered_data.filter(is_relevant)
relevant_data = relevant_data.select(range(min(500, len(relevant_data))))

print(f"Number of relevant examples: {len(relevant_data)}")

# Preprocessing function
max_length = 256
def preprocess_function(example):
    prompt = example["instruction"].strip() + "\nUser: " + example["input"].strip() + "\nAssistant:"
    target = example["output"].strip()
    full_text = prompt + " " + target
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

print("Tokenizing dataset...")
tokenized_dataset = relevant_data.map(preprocess_function, batched=False)

# Training arguments
training_args = TrainingArguments(
    output_dir="./models/response_generator_mentalchat16k",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=1000,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_steps=100,
    fp16=True if torch.cuda.is_available() else False,
    resume_from_checkpoint=True  # Enable checkpoint resuming
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Load model from checkpoint
model = GPT2LMHeadModel.from_pretrained("./models/response_generator_mentalchat16k/checkpoint-750")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

if __name__ == "__main__":
    print("Resuming training from checkpoint-750...")
    trainer.train()
    print("Saving model...")
    trainer.save_model("./models/response_generator_mentalchat16k")
    tokenizer.save_pretrained("./models/response_generator_mentalchat16k")
    print("Training complete. Model saved to ./models/response_generator_mentalchat16k") 