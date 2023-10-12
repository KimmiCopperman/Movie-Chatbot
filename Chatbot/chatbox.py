import random
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging messages

import nltk  
from nltk.stem import WordNetLemmatizer


# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()
# Load the words, classes, and the trained model
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model')

# Load the intents data from intents.json
intents = json.loads(open('intents.json').read())

# Function to clean up and preprocess the input sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


# Function to create a bag of words from the cleaned sentence
def create_bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for i, w in enumerate(words):
        if w in sentence_words:
            bag[i] = 1
    return np.array(bag)

# Function to predict the intent of the user's input
def predict_class(sentence):
    bow = create_bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

# Function to get a response based on the predicted intent
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    
    result = "I'm not sure how to respond to that."  # Default response
    
    for i in list_of_intents:
        if i['tag'] == tag:
            if tag == 'director_movies':
                # If the intent is 'director_movies', fetch the director's movies
                director_name = intents_list[0]['director']
                if director_name in i['responses']:
                    result = ', '.join(i['responses'][director_name])
            else:
                result = random.choice(i['responses'])
            break

    
    return result


print("Hello! This is Theasthai! Your movie maestro!")

while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)
