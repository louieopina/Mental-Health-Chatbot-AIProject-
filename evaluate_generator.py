from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import json
import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK data
nltk.download('punkt')

def load_model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained('./models/response_generator_mentalchat16k')
    tokenizer = AutoTokenizer.from_pretrained('./models/response_generator_mentalchat16k')
    return model, tokenizer

def load_test_data():
    prompts_dataset = load_dataset('json', data_files='data/test_prompts.json')
    answers_dataset = load_dataset('json', data_files='data/ground_truth_answers.json')
    
    # Combine prompts and answers
    test_data = []
    for prompt, answer in zip(prompts_dataset['train'], answers_dataset['train']):
        test_data.append({
            'text': prompt['text'],
            'response': answer['response']
        })
    return test_data

def generate_response(model, tokenizer, text, max_length=100):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def calculate_metrics(predictions, references):
    # Initialize ROUGE scorer
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Calculate ROUGE scores
    rouge_scores = {
        'rouge1': {'precision': [], 'recall': [], 'fmeasure': []},
        'rouge2': {'precision': [], 'recall': [], 'fmeasure': []},
        'rougeL': {'precision': [], 'recall': [], 'fmeasure': []}
    }
    
    # Calculate BLEU scores
    bleu_scores = []
    smoothie = SmoothingFunction().method1
    
    for pred, ref in zip(predictions, references):
        # ROUGE
        scores = rouge.score(ref, pred)
        for metric in ['rouge1', 'rouge2', 'rougeL']:
            rouge_scores[metric]['precision'].append(scores[metric].precision)
            rouge_scores[metric]['recall'].append(scores[metric].recall)
            rouge_scores[metric]['fmeasure'].append(scores[metric].fmeasure)
        
        # BLEU
        reference_tokens = [nltk.word_tokenize(ref)]
        prediction_tokens = nltk.word_tokenize(pred)
        bleu = sentence_bleu(reference_tokens, prediction_tokens, smoothing_function=smoothie)
        bleu_scores.append(bleu)
    
    # Calculate averages
    metrics = {
        'rouge': {
            metric: {
                'precision': np.mean(scores['precision']),
                'recall': np.mean(scores['recall']),
                'fmeasure': np.mean(scores['fmeasure'])
            }
            for metric, scores in rouge_scores.items()
        },
        'bleu': np.mean(bleu_scores)
    }
    
    return metrics

def evaluate_model(model, tokenizer, test_dataset):
    print("Generating responses...")
    predictions = []
    references = []
    
    for item in test_dataset:
        generated = generate_response(model, tokenizer, item['text'])
        predictions.append(generated)
        references.append(item['response'])
    
    print("Calculating metrics...")
    metrics = calculate_metrics(predictions, references)
    
    # Save metrics
    with open('generator_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Generate sample responses
    samples = []
    for i in range(min(10, len(test_dataset))):
        samples.append({
            'input': test_dataset[i]['text'],
            'generated': predictions[i],
            'reference': references[i]
        })
    
    with open('generator_samples.json', 'w') as f:
        json.dump(samples, f, indent=4)
    
    # Plot metrics
    plt.figure(figsize=(12, 6))
    metrics_to_plot = {
        'ROUGE-1 F1': metrics['rouge']['rouge1']['fmeasure'],
        'ROUGE-2 F1': metrics['rouge']['rouge2']['fmeasure'],
        'ROUGE-L F1': metrics['rouge']['rougeL']['fmeasure'],
        'BLEU': metrics['bleu']
    }
    
    plt.bar(metrics_to_plot.keys(), metrics_to_plot.values())
    plt.title('Response Generator Evaluation Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.savefig('generator_metrics.png')
    
    return metrics, samples

def main():
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()
    
    print("Loading test data...")
    test_dataset = load_test_data()
    
    print("Evaluating model...")
    metrics, samples = evaluate_model(model, tokenizer, test_dataset)
    
    print("\nEvaluation Summary:")
    print(f"ROUGE-1 F1: {metrics['rouge']['rouge1']['fmeasure']:.4f}")
    print(f"ROUGE-2 F1: {metrics['rouge']['rouge2']['fmeasure']:.4f}")
    print(f"ROUGE-L F1: {metrics['rouge']['rougeL']['fmeasure']:.4f}")
    print(f"BLEU Score: {metrics['bleu']:.4f}")
    
    print("\nDetailed reports have been saved to:")
    print("- generator_metrics.json")
    print("- generator_samples.json")
    print("- generator_metrics.png")

if __name__ == "__main__":
    main() 