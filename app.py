from flask import Flask, request, jsonify, render_template, session
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
from flask_wtf.csrf import CSRFProtect
import datetime
import re
import random
import unicodedata
from typing import Optional, Dict, Any
import traceback

from config import (
    SESSION_SECRET_KEY,
    SESSION_LIFETIME,
    CACHE_TYPE,
    CACHE_DEFAULT_TIMEOUT,
    RATE_LIMIT,
    MAX_REQUEST_SIZE,
    TIMEOUT,
    CRISIS_KEYWORDS,
    PROFANITY_LIST,
    RESPONSE_TEMPLATES
)
from logger import logger
from model_manager import ModelManager

# Initialize Flask app
app = Flask(__name__)
app.secret_key = SESSION_SECRET_KEY
app.config['PERMANENT_SESSION_LIFETIME'] = SESSION_LIFETIME

# Initialize extensions
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[RATE_LIMIT]
)
cache = Cache(app, config={
    'CACHE_TYPE': CACHE_TYPE,
    'CACHE_DEFAULT_TIMEOUT': CACHE_DEFAULT_TIMEOUT
})
csrf = CSRFProtect(app)

# Initialize model manager
model_manager = ModelManager()

def contains_crisis(text: str) -> bool:
    """Check if the text contains crisis keywords."""
    text = text.lower()
    return any(word in text for word in CRISIS_KEYWORDS)

def contains_profanity(text: str) -> bool:
    """Check if the text contains profanity."""
    text = text.lower()
    return any(word in text for word in PROFANITY_LIST)

def filter_response(text: str) -> str:
    """Filter and clean the response text."""
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "[link removed]", text)
    # Limit to 3 sentences max
    sentences = re.split(r'(?<=[.!?]) +', text)
    text = ' '.join(sentences[:3])
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def humanize_response(text: str) -> str:
    """Add a human touch to the response."""
    text = text.strip()
    if text.lower().startswith("i understand") or text.lower().startswith("i'm sorry"):
        return text
    if len(text.split()) < 12:
        return text + " If you'd like to talk more about it, I'm here to listen."
    if "you should" in text.lower():
        text = text.replace("you should", "maybe you could consider")
    return text

def clean_response(text: str) -> str:
    """Clean the response text."""
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if ord(c) < 128)
    text = text.replace('_', ' ')
    text = ' '.join(text.split())
    return text.strip()

@cache.memoize(timeout=300)
def get_response(user_input: str) -> str:
    """Generate a response for the user input."""
    try:
        # Check for crisis keywords
        if contains_crisis(user_input):
            return random.choice(RESPONSE_TEMPLATES["crisis"])
        
        # Check for profanity
        if contains_profanity(user_input):
            return "I understand you're feeling frustrated, but let's keep our conversation respectful."
        
        # Classify intent
        intent, confidence = model_manager.classify_intent(user_input)
        
        # Generate response
        if confidence > 0.7:
            response = model_manager.generate_response(user_input)
            if response:
                return humanize_response(clean_response(response))
        
        # Fallback to template response
        return random.choice(RESPONSE_TEMPLATES.get(intent, RESPONSE_TEMPLATES["general"]))
    
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}\n{traceback.format_exc()}")
        return "I'm having trouble processing your message right now. Please try again in a moment."

@app.route("/")
def index():
    """Render the main page."""
    return render_template("index.html")

@csrf.exempt
@app.route("/chat", methods=["POST"])
@limiter.limit("10/minute")
def chat():
    """Handle chat requests."""
    try:
        user_input = request.json.get("message", "").strip()
        
        # Validate input
        if not user_input:
            return jsonify({"error": "Message cannot be empty"}), 400
        
        if len(user_input.encode('utf-8')) > MAX_REQUEST_SIZE:
            return jsonify({"error": "Message too long"}), 400
        
        # Generate response
        response = get_response(user_input)
        
        # Log interaction
        logger.info(f"User: {user_input}")
        logger.info(f"Bot: {response}")
        
        return jsonify({"response": response})
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    try:
        # Check if models are loaded
        if not model_manager.intent_classifier or not model_manager.response_generator:
            return jsonify({"status": "error", "message": "Models not loaded"}), 503
        
        return jsonify({"status": "healthy"}), 200
    
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 503

@csrf.exempt
@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    """Clear the chat session."""
    session.clear()
    return jsonify({"status": "success"})

if __name__ == "__main__":
    # Warm up models
    model_manager.warm_up()
    
    # Start the application
    app.run(host='0.0.0.0', port=5000, debug=False) 