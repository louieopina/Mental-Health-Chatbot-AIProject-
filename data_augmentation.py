import json
import random
from collections import Counter
from typing import List, Dict, Tuple
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from transformers import pipeline
import re
from difflib import SequenceMatcher

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Load sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

def load_data(file_path: str) -> List[Dict]:
    """Load data from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_data(data: List[Dict], file_path: str):
    """Save data to JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def get_synonym(word: str) -> str:
    """Get a random synonym for a word."""
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.name() != word:
                synonyms.append(lemma.name())
    return random.choice(synonyms) if synonyms else word

def synonym_replacement(text: str, n: int = 1) -> str:
    """Replace n random words with their synonyms."""
    words = word_tokenize(text)
    n = min(n, len(words))
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word.isalnum()]))
    random.shuffle(random_word_list)
    num_replaced = 0
    
    for random_word in random_word_list:
        synonyms = get_synonym(random_word)
        if synonyms != random_word:
            new_words = [synonyms if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    
    return ' '.join(new_words)

def random_deletion(text: str, p: float = 0.1) -> str:
    """Randomly delete words with probability p."""
    words = text.split()
    if len(words) <= 3:
        return text
    
    new_words = []
    for word in words:
        if random.random() > p:
            new_words.append(word)
    
    return ' '.join(new_words) if new_words else text

def random_swap(text: str, n: int = 1) -> str:
    """Randomly swap n pairs of adjacent words."""
    words = text.split()
    n = min(n, len(words) - 1)
    new_words = words.copy()
    
    for _ in range(n):
        idx = random.randint(0, len(new_words) - 2)
        new_words[idx], new_words[idx + 1] = new_words[idx + 1], new_words[idx]
    
    return ' '.join(new_words)

def back_translation(text: str) -> str:
    """Perform back translation using word substitutions."""
    # Enhanced word substitutions with context
    substitutions = {
        'happy': ['joyful', 'delighted', 'cheerful'],
        'sad': ['unhappy', 'downcast', 'melancholy'],
        'angry': ['furious', 'enraged', 'irate'],
        'tired': ['exhausted', 'weary', 'fatigued'],
        'worried': ['anxious', 'concerned', 'distressed'],
        'scared': ['frightened', 'terrified', 'afraid'],
        'excited': ['thrilled', 'elated', 'enthusiastic'],
        'nervous': ['anxious', 'jittery', 'tense'],
        'depressed': ['down', 'gloomy', 'dejected'],
        'anxious': ['worried', 'nervous', 'apprehensive'],
        'stress': ['pressure', 'tension', 'strain'],
        'overwhelmed': ['swamped', 'overloaded', 'burdened'],
        'lonely': ['isolated', 'solitary', 'alone'],
        'confused': ['perplexed', 'bewildered', 'puzzled'],
        'frustrated': ['annoyed', 'irritated', 'exasperated']
    }
    
    words = text.split()
    new_words = []
    
    for word in words:
        word_lower = word.lower()
        if word_lower in substitutions:
            new_word = random.choice(substitutions[word_lower])
            new_words.append(new_word)
        else:
            new_words.append(word)
    
    return ' '.join(new_words)

def check_sentiment_similarity(original: str, augmented: str) -> bool:
    """Check if the sentiment of the augmented text is similar to the original."""
    orig_sentiment = sentiment_analyzer(original)[0]
    aug_sentiment = sentiment_analyzer(augmented)[0]
    
    # Check if the sentiment label is the same
    if orig_sentiment['label'] != aug_sentiment['label']:
        return False
    
    # Check if the sentiment scores are within 0.2 of each other
    return abs(orig_sentiment['score'] - aug_sentiment['score']) < 0.2

def check_similarity(original: str, augmented: str) -> float:
    """Check the similarity between original and augmented text."""
    return SequenceMatcher(None, original, augmented).ratio()

def augment_text(text: str) -> List[str]:
    """Generate augmented versions of the input text with validation."""
    augmented_texts = []
    
    # Try different augmentation techniques
    techniques = [
        lambda t: synonym_replacement(t, n=2),
        lambda t: random_deletion(t, p=0.1),
        lambda t: random_swap(t, n=1),
        lambda t: back_translation(t)
    ]
    
    for technique in techniques:
        try:
            augmented = technique(text)
            
            # Validate the augmented text
            if (check_similarity(text, augmented) > 0.7 and  # Not too different
                check_sentiment_similarity(text, augmented) and  # Similar sentiment
                len(augmented.split()) >= 3):  # Not too short
                augmented_texts.append(augmented)
        except Exception as e:
            print(f"Error in augmentation: {e}")
            continue
    
    return augmented_texts

def balance_dataset(data: List[Dict], target_samples: int = 100) -> List[Dict]:
    """Balance the dataset by augmenting underrepresented classes."""
    # Count current distribution
    intent_counts = Counter(item['intent'] for item in data)
    
    # Find the maximum count
    max_count = max(intent_counts.values())
    
    # Augment data for underrepresented classes
    augmented_data = data.copy()
    
    for intent, count in intent_counts.items():
        if count < target_samples:
            # Get samples of this intent
            intent_samples = [item for item in data if item['intent'] == intent]
            
            # Calculate how many new samples we need
            samples_needed = target_samples - count
            
            # Generate new samples
            attempts = 0
            max_attempts = samples_needed * 3  # Allow for some failed attempts
            
            while len([item for item in augmented_data if item['intent'] == intent]) < target_samples and attempts < max_attempts:
                # Randomly select a sample to augment
                sample = random.choice(intent_samples)
                
                # Generate augmented text
                augmented_texts = augment_text(sample['text'])
                
                # Add new samples
                for aug_text in augmented_texts:
                    if len([item for item in augmented_data if item['intent'] == intent]) >= target_samples:
                        break
                    
                    new_sample = {
                        'text': aug_text,
                        'intent': intent
                    }
                    augmented_data.append(new_sample)
                
                attempts += 1
    
    return augmented_data

def main():
    # Load the original dataset
    data = load_data('data/test_intent.json')
    
    # Print original distribution
    original_counts = Counter(item['intent'] for item in data)
    print("\nOriginal intent distribution:")
    for intent, count in original_counts.items():
        print(f"{intent}: {count}")
    
    # Balance the dataset
    balanced_data = balance_dataset(data)
    
    # Save the balanced dataset
    save_data(balanced_data, 'data/balanced_intent.json')
    
    # Print the new distribution
    new_intent_counts = Counter(item['intent'] for item in balanced_data)
    print("\nNew intent distribution:")
    for intent, count in new_intent_counts.items():
        print(f"{intent}: {count}")

if __name__ == "__main__":
    main() 