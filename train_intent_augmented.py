from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
import json
import random
from sklearn.metrics import classification_report
import argparse

# Add new argument for resuming training
parser = argparse.ArgumentParser(description='Train an intent classifier')
parser.add_argument('--resume_from_checkpoint', action='store_true', help='Resume training from the latest checkpoint')

def load_augmented_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    random.shuffle(data)
    texts = [ex['text'] for ex in data]
    intents = [ex['intent'] for ex in data]
    return texts, intents

def prepare_dataset():
    texts, intents = load_augmented_dataset('augmented_intent_dataset.jsonl')
    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(intents)
    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
    # Split train/test (80/20)
    n = len(texts)
    split = int(n * 0.8)
    train_encodings = {k: v[:split] for k, v in encodings.items()}
    test_encodings = {k: v[split:] for k, v in encodings.items()}
    train_labels = labels[:split]
    test_labels = labels[split:]
    class IntentDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item
        def __len__(self):
            return len(self.labels)
    train_dataset = IntentDataset(train_encodings, train_labels)
    test_dataset = IntentDataset(test_encodings, test_labels)
    return train_dataset, test_dataset, len(label_encoder.classes_), label_encoder

def train_intent_classifier(resume_from_checkpoint=False):
    train_dataset, test_dataset, num_labels, label_encoder = prepare_dataset()
    if resume_from_checkpoint:
        model = AutoModelForSequenceClassification.from_pretrained('./intent_classifier_augmented/checkpoint-1210')
    else:
        model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)
    training_args = TrainingArguments(
        output_dir='./intent_classifier_augmented',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy'
    )
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = (predictions == labels).mean()
        return {'accuracy': accuracy}
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    print('\nStarting training...')
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model('./intent_classifier_augmented')
    print('\nModel saved to ./intent_classifier_augmented')
    print('\nEvaluating model...')
    eval_results = trainer.evaluate()
    print(f'Evaluation results: {eval_results}')
    # Save label encoder
    with open('intent_label_encoder.json', 'w', encoding='utf-8') as f:
        json.dump({'classes': label_encoder.classes_.tolist()}, f)
    print('Label encoder saved as intent_label_encoder.json')

    # Compute F1-score and per-class metrics
    print('\nComputing F1-score and per-class metrics...')
    # Get predictions on test set
    predictions = trainer.predict(test_dataset)
    y_true = predictions.label_ids
    y_pred = np.argmax(predictions.predictions, axis=1)
    report = classification_report(y_true, y_pred, target_names=label_encoder.classes_, digits=4)
    print(report)
    with open('f1_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print('F1-score report saved as f1_report.txt')

if __name__ == '__main__':
    args = parser.parse_args()
    train_intent_classifier(resume_from_checkpoint=args.resume_from_checkpoint) 