from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
from preprocess import preprocess_dataset

def prepare_dataset_for_intent_classification():
    # Load the dataset
    dataset = load_dataset("heliosbrahma/mental_health_chatbot_dataset")
    
    # Create a simple intent classification based on text content
    def get_intent(example):
        text = example["text"].lower()
        if "crisis" in text or "emergency" in text or "suicide" in text:
            return "crisis"
        elif "advice" in text or "help" in text or "suggestion" in text:
            return "advice"
        elif "question" in text or "ask" in text or "what" in text:
            return "question"
        elif "thank" in text or "grateful" in text:
            return "gratitude"
        else:
            return "general"
    
    # Add intent labels
    dataset_with_intents = dataset.map(lambda x: {"intent": get_intent(x)})
    
    # Convert string labels to integers
    label_encoder = LabelEncoder()
    intents = dataset_with_intents["train"]["intent"]
    label_encoder.fit(intents)
    
    def encode_labels(example):
        example["labels"] = label_encoder.transform([example["intent"]])[0]
        return example
    
    # Apply label encoding
    final_dataset = dataset_with_intents.map(encode_labels)
    
    # Tokenize the text
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    def tokenize_function(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=128,
            padding="max_length"
        )
    tokenized_dataset = final_dataset.map(tokenize_function, batched=False)
    
    # Split into train and validation sets (80-20 split)
    train_val_split = tokenized_dataset["train"].train_test_split(test_size=0.2)
    
    return train_val_split, len(label_encoder.classes_)

def train_intent_classifier():
    # Prepare dataset
    dataset, num_labels = prepare_dataset_for_intent_classification()
    
    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./intent_classifier",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )
    
    # Define compute_metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics
    )
    
    # Train the model
    print("\nStarting training...")
    trainer.train()
    
    # Save the model
    trainer.save_model("./intent_classifier")
    print("\nModel saved to ./intent_classifier")
    
    # Evaluate the model
    print("\nEvaluating model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

if __name__ == "__main__":
    train_intent_classifier() 