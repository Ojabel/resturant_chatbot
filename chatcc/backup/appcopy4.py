import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
from textblob import TextBlob
import os
import subprocess

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with your own secret key

lemmatizer = WordNetLemmatizer()

# Load the model and other necessary files
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Store conversation context to improve engagement
conversation_history = {}

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json, user_id):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    response = "I'm not sure how to respond to that."
    
    # Check if there's a conversation history to enhance the response
    if user_id in conversation_history:
        context = conversation_history[user_id]
        if context.get('last_intent') == tag:
            response = f"Hi again! It seems you're asking about '{tag}'. Let me help you further!"
    
    for i in list_of_intents:
        if i['tag'] == tag:
            # Randomly select a response from the list of responses for the intent
            response = random.choice(i['responses'])
            break
    
    # Update conversation history with the current intent
    conversation_history[user_id] = {'last_intent': tag}
    
    return response

def get_follow_up():
    follow_up_questions = [
        "Is there anything else I can help you with?",
        "Do you need any further clarification?",
        "Feel free to ask me more questions!",
        "Would you like to know more about this topic?",
        "Can I assist you with anything else?"
    ]
    return random.choice(follow_up_questions)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'password':  # Replace with your own credentials
            session['logged_in'] = True
            return redirect(url_for('admin'))
        else:
            flash('Invalid credentials. Please try again.')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/admin')
def admin():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('admin.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get("message")
        user_id = request.json.get("user_id", "default_user")  # Using user_id to track conversation context
        if not user_message:
            return jsonify({"response": "Message not provided."}), 400

        if model is None or intents is None or words is None or classes is None:
            return jsonify({"response": "Server resources not properly loaded. Please try again later."}), 500

        # Sentiment analysis
        sentiment = TextBlob(user_message).sentiment.polarity
        sentiment_response = ""

        if sentiment > 0:
            sentiment_response = "I'm glad you're feeling positive! 😊"
        elif sentiment < 0:
            sentiment_response = "I'm sorry to hear that you're feeling down. 😔 How can I help you?"
        else:
            sentiment_response = "Let's keep going! 😊"

        ints = predict_class(user_message, model)
        if not ints:
            handle_unrecognized_message(user_message)
            return jsonify({"response": "Hmm, I didn't quite understand that. Could you rephrase?"})

        # Get the bot's response
        response = get_response(ints, intents, user_id)

        # Get a random follow-up question
        follow_up = get_follow_up()
        
        # Combine the sentiment response, the main response, and the follow-up question
        full_response = f"{sentiment_response} {response} {follow_up}"
        return jsonify({"response": full_response})
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({"response": "Oops! Something went wrong. Please try again later."}), 500

def handle_unrecognized_message(message):
    try:
        file_path = 'unrecognized_messages.json'
        print(f"Handling unrecognized message: {message}")
        if os.path.exists(file_path):
            with open(file_path, 'r+') as file:
                data = json.load(file)
                for intent in data['intents']:
                    if message in intent['patterns']:
                        print(f"Message '{message}' already exists in unrecognized intents.")
                        return
                new_entry = {"tag": "unrecognized", "patterns": [message], "responses": [""]}
                data['intents'].append(new_entry)
                file.seek(0)
                json.dump(data, file, indent=4)
                print(f"Message '{message}' added to unrecognized intents.")
        else:
            with open(file_path, 'w') as file:
                data = {"intents": [{"tag": "unrecognized", "patterns": [message], "responses": [""]}]}
                json.dump(data, file, indent=4)
                print(f"File '{file_path}' created and message '{message}' added to unrecognized intents.")
    except Exception as e:
        print(f"Error in handle_unrecognized_message: {e}")

@app.route('/log_unrecognized', methods=['POST'])
def log_unrecognized():
    try:
        message = request.json.get('message')
        if not message:
            return jsonify({"status": "Message not provided."}), 400
        handle_unrecognized_message(message)
        return jsonify({"status": "Logged"})
    except Exception as e:
        print(f"Error in log_unrecognized endpoint: {e}")
        return jsonify({"status": "An internal error occurred. Please try again later."}), 500

@app.route('/get_unknown_intents', methods=['GET'])
def get_unknown_intents():
    try:
        file_path = 'unrecognized_messages.json'
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                data = json.load(file)
                return jsonify(data)
        else:
            return jsonify({"intents": []})
    except Exception as e:
        print(f"Error in get_unknown_intents endpoint: {e}")
        return jsonify({"status": "An internal error occurred. Please try again later."}), 500

@app.route('/update_intents', methods=['POST'])
def update_intents():
    try:
        new_intents = request.json.get('intents')
        if not new_intents:
            return jsonify({"status": "No intents provided."}), 400

        with open('intents.json', 'r+') as file:
            data = json.load(file)
            data['intents'].extend(new_intents)
            file.seek(0)
            json.dump(data, file, indent=4)

        if os.path.exists('unrecognized_messages.json'):
            with open('unrecognized_messages.json', 'r+') as file:
                unrecognized_data = json.load(file)
                updated_patterns = [pattern for intent in new_intents for pattern in intent['patterns']]
                unrecognized_data['intents'] = [
                    intent for intent in unrecognized_data['intents'] if not any(pattern in updated_patterns for pattern in intent['patterns'])]
                file.seek(0)
                file.truncate()
                json.dump(unrecognized_data, file, indent=4)

        return jsonify({"status": "Intents updated and unrecognized messages cleaned."})
    except Exception as e:
        print(f"Error in update_intents endpoint: {e}")
        return jsonify({"status": "An internal error occurred. Please try again later."}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        result = subprocess.run(['python', 'train_chatbot.py'], capture_output=True, text=True)
        return jsonify({"status": "success", "output": result.stdout})
    except Exception as e:
        print(f"Error in retrain endpoint: {e}")
        return jsonify({"status": "An internal error occurred. Please try again later."}), 500

if __name__ == '__main__':
    app.run(debug=True)
