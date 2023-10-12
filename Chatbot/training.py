import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
import json  
import random

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the intents data from intents.json
intents = json.loads(open('intents.json').read())

# Rest of your code...


# Initialize lists and variables
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Extract words, classes, and documents from intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and clean the words
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

# Save words and classes for future use
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Prepare training data
training_data = []

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = [0] * len(classes)
    output_row[classes.index(document[1])] = 1
    training_data.append(bag + output_row)

# Shuffle and convert training data
random.shuffle(training_data)
training_data = np.array(training_data)
pickle.dump(training_data, open('training_data.pkl', 'wb'))

# Separate training data into features (train_x) and labels (train_y)
train_x = training_data[:, :len(words)]
train_y = training_data[:, len(words):]

# Define and compile the neural network model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))

sgd = tf.keras.optimizers.legacy.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
history = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save the model to a file
model.save('chatbot_model', save_format='tf')

print("Happy Watching!")
