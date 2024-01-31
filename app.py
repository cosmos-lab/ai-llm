import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from flask import Flask, request, jsonify

app = Flask(__name__)

# Define global variables for tokenizer and max_sequence_length
tokenizer = None
max_sequence_length = None

# Function to train and save the model
def train_and_save_model():
    global tokenizer, max_sequence_length

    # Sample story as text_data
    text_data = """
    Once upon a time in a magical land, there lived a wise old wizard named Merlin.
    Merlin was known for his extraordinary powers and the ability to answer any question.
    People from far and wide would seek Merlin's advice, hoping to gain knowledge and wisdom.

    One day, a curious traveler approached Merlin and asked, "What is the secret to true happiness?"
    Merlin, with a twinkle in his eye, replied, "The key to true happiness lies in gratitude and kindness.
    Appreciate the beauty around you, and let kindness be your guiding light."

    The traveler thanked Merlin for his wise words and continued on his journey, carrying the newfound wisdom in his heart.
    And so, the legend of Merlin, the wise wizard, and his timeless teachings spread across the magical land.
    """

    # Tokenize the text
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts([text_data])
    total_words = len(tokenizer.word_index) + 1

    # Create input sequences and labels
    input_sequences = []
    for line in text_data.split('\n'):
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    max_sequence_length = max([len(seq) for seq in input_sequences])
    input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')
    X, y = input_sequences[:, :-1], input_sequences[:, -1]
    y = tf.keras.utils.to_categorical(y, num_classes=total_words)

    # Build the model
    model = Sequential()
    model.add(Embedding(total_words, 10000, input_length=max_sequence_length-1))
    model.add(LSTM(100))
    model.add(Dense(total_words, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X, y, epochs=200, verbose=2)

    # Save the model in the native Keras format
    model.save('text_generation_model', save_format='tf')

# API endpoint for question answering
@app.route('/answer_question', methods=['GET'])
def answer_question():
    try:
        input_question = request.args.get('input_question', '')
        next_words = int(request.args.get('next_words', 10))  # Default to 10 words if 'next_words' not provided

        # Load the trained model
        model = load_model('text_generation_model')

        seed_text = input_question
        generated_text = input_question  # Initialize with the original input

        for _ in range(next_words):
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
            predicted_probabilities = model.predict(token_list, verbose=0)
            predicted_class = np.argmax(predicted_probabilities)
            output_word = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted_class:
                    output_word = word
                    break
            seed_text += " " + output_word
            generated_text += " " + output_word

        return jsonify({'answer': generated_text})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    train_and_save_model()
    app.run(host='0.0.0.0', port=5000, debug=True)
