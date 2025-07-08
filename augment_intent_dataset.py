import json
import random
from nltk.corpus import wordnet
import nltk
nltk.download('wordnet')

def synonym_replacement(text, n=2):
    words = text.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if len(word) > 3]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = set()
        for syn in wordnet.synsets(random_word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ').lower()
                if synonym != random_word and synonym.isalpha():
                    synonyms.add(synonym)
        if len(synonyms) > 0:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    return ' '.join(new_words)

def augment_dataset(input_path, output_path, aug_per_example=1):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    augmented = []
    for ex in data:
        augmented.append(ex)
        for _ in range(aug_per_example):
            aug_text = synonym_replacement(ex['text'])
            if aug_text != ex['text']:
                augmented.append({'text': aug_text, 'intent': ex['intent']})
    random.shuffle(augmented)
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in augmented:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    print(f"Augmented dataset saved as {output_path} with {len(augmented)} examples.")

if __name__ == "__main__":
    augment_dataset('combined_intent_dataset.jsonl', 'augmented_intent_dataset.jsonl', aug_per_example=1) 