import os
from typing import List, Dict

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths
INTENT_CLASSIFIER_PATH = os.path.join(BASE_DIR, "intent_classifier_augmented")
RESPONSE_GENERATOR_PATH = os.path.join(BASE_DIR, "models", "response_generator_mentalchat16k")

# Model settings
MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 1
DEVICE = "cuda" if os.environ.get("USE_GPU", "false").lower() == "true" else "cpu"

# API settings
RATE_LIMIT = "100/hour"  # Requests per hour
MAX_REQUEST_SIZE = 1024  # Maximum request size in bytes
TIMEOUT = 30  # Request timeout in seconds

# Crisis detection
CRISIS_KEYWORDS: List[str] = [
    "suicide", "kill myself", "end my life", "self-harm", "emergency",
    "crisis", "hurt myself", "can't go on", "give up", "die",
    "depressed", "hopeless"
]

# Content filtering
PROFANITY_LIST: List[str] = [
    "damn", "shit", "fuck", "bitch", "bastard", "asshole",
    "dick", "crap", "piss", "cunt"
]

# Response templates
RESPONSE_TEMPLATES: Dict[str, List[str]] = {
    "crisis": [
        "I'm really sorry you're feeling this way. Your feelings are valid and important. "
        "If you're thinking about harming yourself, please reach out to someone you trust or call 116 123 (Samaritans, UK, free 24/7). "
        "You're not alone, and there are people who care about you.",
        # ... add more templates
    ],
    # ... add more intent templates
    "general": [
        "Thank you for sharing. I'm here to listen and support you.",
        "I'm here for you. Would you like to talk more about what's on your mind?",
        "It's okay to feel this way. If you'd like to share more, I'm here.",
        "I'm here to support you. Please tell me more if you'd like.",
        "You are not alone. I'm here to listen whenever you need."
    ]
}

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = os.path.join(BASE_DIR, "logs", "app.log")

# Session settings
SESSION_SECRET_KEY = os.environ.get("SESSION_SECRET_KEY", "your-secret-key-here")
SESSION_LIFETIME = 3600  # 1 hour in seconds

# Cache settings
CACHE_TYPE = "simple"
CACHE_DEFAULT_TIMEOUT = 300  # 5 minutes

# Health check settings
HEALTH_CHECK_INTERVAL = 60  # seconds
HEALTH_CHECK_TIMEOUT = 5  # seconds 