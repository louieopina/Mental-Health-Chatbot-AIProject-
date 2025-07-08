# AI-Powered Mental Health Chatbot

## Overview
This project is an advanced AI-powered mental health chatbot designed to provide empathetic, context-aware support for users experiencing a range of mental health concerns. Leveraging state-of-the-art transformer models (DistilBERT for intent classification and GPT-2 for response generation), the chatbot delivers safe, relevant, and supportive responses while ensuring robust crisis detection and ethical safeguards.

## Features
- Intent classification using DistilBERT
- Response generation using fine-tuned GPT-2
- Crisis detection and escalation protocols
- Ethical safeguards and privacy protection
- Web-based user interface (Flask)
- Comprehensive evaluation metrics and visualisations

## Directory Structure
```
ProjectAI/
│
├── app.py                        # Main Flask app
├── requirements.txt              # Python dependencies
├── technical_report.md           # Final technical report
├── README.md                     # This file
│
├── models/
│   ├── intent_classifier_augmented/         # Trained intent classifier
│   └── response_generator_mentalchat16k/    # Trained response generator
│
├── data/
│   ├── augmented_intent_dataset.jsonl       # Augmented/combined dataset
│   ├── test_prompts.json                    # Test prompts
│   ├── generated_responses.json             # Generated responses
│   └── ground_truth_answers.json            # Ground truth for evaluation
│
├── logs/
│   └── f1_report.txt                        # Latest F1-score report
│
├── templates/
│   └── index.html                           # Web UI template
│
├── generate_visualizations.py               # Script to generate all figures
├── train_intent_augmented.py                # Training script for intent classifier
├── train_generator.py                       # Training script for response generator
├── augment_intent_dataset.py                # Data augmentation script
├── combine_and_label_intent_dataset.py      # Data combining script
```

## Setup Instructions
1. **Navigate to the project directory on your machine.**
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Ensure models and data are in the correct directories.**
4. **Run the Flask web app:**
   ```sh
   python app.py web
   ```
5. **Open your browser and go to:**
   http://127.0.0.1:5000

## Usage
- Interact with the chatbot via the web interface.
- For testing and evaluation, use the provided test prompts and scripts.

## Datasets
This project uses two primary datasets for training and evaluation:

- **MentalChat16K** ([ShenLab/MentalChat16K](https://huggingface.co/datasets/ShenLab/MentalChat16K))
  - 16,000 conversations
  - Multiple mental health domains
  - Professional annotations
  - Quality-controlled responses

- **Heliosbrahma Mental Health Chatbot Dataset** ([heliosbrahma/mental_health_chatbot_dataset](https://huggingface.co/datasets/heliosbrahma/mental_health_chatbot_dataset))
  - 10,000 conversations
  - Diverse user intents
  - Emotional context labels
  - Response quality metrics

These datasets were combined, cleaned, and balanced to create the training and evaluation sets for both intent classification and response generation.

## Evaluation
- The system is evaluated using accuracy, precision, recall, F1-score, and per-class metrics.
- **Intent Classifier (DistilBERT):**
  - Best Evaluation Accuracy: **0.91** (91%)
  - F1-score: *Not available or not reported in current logs*
- Visualisations are available in the `figures/` directory and referenced in the technical report.

## Ethical Considerations
- The chatbot is not a substitute for professional help.
- Crisis detection and escalation protocols are in place.
- All user data is anonymised and handled with strict privacy safeguards.

## Contact
For questions or support, please contact:
- Louie Opina
- louie.opina@yahoo.com
