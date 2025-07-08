from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def load_model_and_tokenizer():
    model = AutoModelForSequenceClassification.from_pretrained('intent_classifier_augmented')
    tokenizer = AutoTokenizer.from_pretrained('intent_classifier_augmented')
    return model, tokenizer

def load_test_data():
    dataset = load_dataset('json', data_files='data/test_intent.json')
    test_dataset = dataset['train']
    return test_dataset

def evaluate_model(model, tokenizer, test_dataset):
    # Load label encoder
    with open('models/intent_label_encoder.json', 'r') as f:
        label_encoder = json.load(f)
    
    # Ensure all labels are in the encoder
    all_labels = set(test_dataset['intent'])
    for label in all_labels:
        if label not in label_encoder:
            label_encoder[label] = len(label_encoder)
    
    # Prepare test data
    texts = test_dataset['text']
    true_labels = test_dataset['intent']
    
    # Convert labels to numeric
    true_labels = [label_encoder[label] for label in true_labels]
    
    # Tokenize and predict
    predictions = []
    for i, text in enumerate(texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        logits = outputs.logits
        pred = logits.argmax(-1).item()
        predictions.append(pred)
        print(f"Sample {i}: True Label: {true_labels[i]}, Predicted Label: {pred}, Logits: {logits.detach().numpy()}")
    
    # Generate reports
    class_names = list(label_encoder.keys())
    
    # Classification report
    report = classification_report(true_labels, predictions, 
                                 target_names=class_names, 
                                 output_dict=True)
    
    # Save classification report
    with open('intent_classification_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    # Generate confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('intent_confusion_matrix.png')
    
    # Calculate per-class metrics
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            'precision': report[class_name]['precision'],
            'recall': report[class_name]['recall'],
            'f1-score': report[class_name]['f1-score'],
            'support': report[class_name]['support']
        }
    
    # Save per-class metrics
    with open('intent_per_class_metrics.json', 'w') as f:
        json.dump(per_class_metrics, f, indent=4)
    
    # Generate summary report
    summary = {
        'overall_accuracy': report['accuracy'],
        'macro_avg_f1': report['macro avg']['f1-score'],
        'weighted_avg_f1': report['weighted avg']['f1-score'],
        'total_samples': len(true_labels)
    }
    
    with open('intent_evaluation_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    return report, per_class_metrics, summary

def main():
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()
    
    print("Loading test data...")
    test_dataset = load_test_data()
    
    print("Evaluating model...")
    report, per_class_metrics, summary = evaluate_model(model, tokenizer, test_dataset)
    
    print("\nEvaluation Summary:")
    print(f"Overall Accuracy: {summary['overall_accuracy']:.4f}")
    print(f"Macro Average F1: {summary['macro_avg_f1']:.4f}")
    print(f"Weighted Average F1: {summary['weighted_avg_f1']:.4f}")
    print(f"Total Test Samples: {summary['total_samples']}")
    
    print("\nDetailed reports have been saved to:")
    print("- intent_classification_report.json")
    print("- intent_per_class_metrics.json")
    print("- intent_evaluation_summary.json")
    print("- intent_confusion_matrix.png")

if __name__ == "__main__":
    main() 