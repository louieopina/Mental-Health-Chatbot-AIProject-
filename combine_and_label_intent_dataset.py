import json
from datasets import load_dataset, Dataset, concatenate_datasets
import re
from collections import Counter
import random

def get_intent(text):
    text = text.lower()
    if any(word in text for word in ["crisis", "emergency", "suicide", "kill myself", "end my life", "self-harm", "can't go on", "give up", "die", "hopeless"]):
        return "crisis"
    if any(word in text for word in ["flashback", "trauma", "reliving", "memories come back", "ptsd"]):
        return "trauma"
    if any(word in text for word in ["motivation", "can't find the motivation", "break out of this cycle"]):
        return "motivation"
    if any(word in text for word in ["not good enough", "believe in myself", "self-esteem"]):
        return "self_esteem"
    if any(word in text for word in ["anxious", "anxiety", "uncertainty", "future", "graduating"]):
        return "anxiety"
    if any(word in text for word in ["friends don't care", "feel alone", "lonely", "no one cares"]):
        return "loneliness"
    if any(word in text for word in ["guilt", "blaming myself", "passed away", "grief"]):
        return "grief"
    if any(word in text for word in ["arguing", "partner", "relationship", "communicate"]):
        return "relationship"
    if any(word in text for word in ["work", "burnout", "exhausted", "boss", "break", "help"]):
        return "burnout"
    if any(word in text for word in ["nervous", "social situations", "comfortable around others", "social anxiety"]):
        return "social_anxiety"
    return "general"

def clean_text(text):
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_and_label():
    # Load both datasets
    ds1 = load_dataset("heliosbrahma/mental_health_chatbot_dataset")
    ds2 = load_dataset("ShenLab/MentalChat16K")
    # Extract text fields, skip non-string or empty
    ds1_texts = [clean_text(x["text"]) for x in ds1["train"] if isinstance(x["text"], str) and x["text"].strip()]
    ds2_texts = [clean_text(x["input"]) for x in ds2["train"] if isinstance(x["input"], str) and x["input"].strip()]
    # Assign intents
    ds1_labeled = [{"text": t, "intent": get_intent(t)} for t in ds1_texts]
    ds2_labeled = [{"text": t, "intent": get_intent(t)} for t in ds2_texts]
    # Combine
    combined = ds1_labeled + ds2_labeled
    # Deduplicate
    seen = set()
    deduped = []
    for ex in combined:
        key = (ex["text"].lower(), ex["intent"])
        if key not in seen:
            seen.add(key)
            deduped.append(ex)
    # Balance classes (optional: limit to max 2000 per class)
    class_counts = Counter([ex["intent"] for ex in deduped])
    max_per_class = 2000
    balanced = []
    per_class = {k: 0 for k in class_counts}
    random.shuffle(deduped)
    for ex in deduped:
        intent = ex["intent"]
        if per_class[intent] < max_per_class:
            balanced.append(ex)
            per_class[intent] += 1
    # Save as JSONL
    with open("combined_intent_dataset.jsonl", "w", encoding="utf-8") as f:
        for ex in balanced:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Combined, cleaned, and balanced dataset saved as combined_intent_dataset.jsonl with {len(balanced)} examples.")

if __name__ == "__main__":
    load_and_label() 