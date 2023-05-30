import os
import csv
import random
import numpy as np
import tensorflow as tf
from typing import List, Dict
from keras.utils import pad_sequences
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model


# Method that reads the notes from a csv.
def read_notes_csv(fname: str, delimiter=';') -> List:
    rows = list()
    with open(fname) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delimiter, quoting=csv.QUOTE_NONNUMERIC)
        for row in csv_reader:
            rows.append(row)
    return rows


# Method that transforms the notes (list of ints) into words (string).
def notes_to_word(notes: List) -> List:
    notes_words = list()
    for note in notes:
        if isinstance(note, list):
            word = "".join('{:0>3}'.format(int(n)) for n in note)
        notes_words.append(word)
    return notes_words


# Method that creates the vocab dictionary and fills it.
def create_vocab(notes_word: List) -> Dict:
    vocab, index = {}, 1  # start indexing from 1
    vocab['<pad>'] = 0  # add a padding token
    for note in notes_word:
        if note not in vocab:
            vocab[note] = index
            index += 1
    return vocab


# Method that generates the sequences.
def generate_sequences(notes_words: List, sequence_size: int, ovelap_size: int = 1) -> List:
    return [notes_words[i:i + sequence_size] for i in range(0, len(notes_words), sequence_size - ovelap_size)]


# Method that generates n-gram based on window size, number of negative samples and vocabulary size.
def generate_n_grams(notes_list: List, max_len:int=4):
    input_sequences = list()
    to = 1
    ifrom = 0
    for i in range(1, len(notes_list)):
        if i >= max_len:
            ifrom += 1
        if to < max_len:
            to += 1
        else:
            to = ifrom + max_len
        n_gram_seqs = notes_list[ifrom:to]
        input_sequences.append(n_gram_seqs)
    return input_sequences


# Method that pads the n-grams to make them all have the same size.
def pad_n_grams(notes_list: List):
    max_seq_length = max([len(x) for x in notes_list])
    return np.array(pad_sequences(notes_list, maxlen=max_seq_length, padding='pre'))


# Method that creates the model to use later.
def create_model(vocab_size, embedding_dim, embeddings_matrix, max_seq_length):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embeddings_matrix],
                                      input_length=max_seq_length - 1, trainable=False),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(vocab_size, activation='softmax')
        ]
    )
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    return model


# Reverse method of notes_to_words.
def words_to_notes(notes_words):
    segments_lists = [[note_word[i:i + 3] for i in range(0, len(note_word), 3)] for note_word in notes_words]
    notes = [[int(segment) for segment in segments] for segments in segments_lists]
    return notes

# method that does all the preprocessing needed for the training and prediction, separated for modularity.
def prepare_data(notes, max_seq_length, embedding_dim):
    notas = read_notes_csv(notes)
    notes_word = notes_to_word(notas)  # ['048055063070074000000000', '048060063067070000000000', ...]
    assert len(notes_word) > 0

    vocab = create_vocab(notes_word)
    vocab_size = len(vocab)

    encode_notes_words = [vocab[word] for word in notes_word]  # [1, 2, 3, 4, 1]

    input_sequences = generate_n_grams(encode_notes_words, max_len=max_seq_length)
    padded_sequences = pad_n_grams(input_sequences)

    x_values, labels = padded_sequences[:, :-1], padded_sequences[:, -1]
    y_values = tf.keras.utils.to_categorical(labels)

    embeddings_index = {}
    with open('embedding_vectors.csv') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coeffs = np.array(values[1:], dtype='float32')
            embeddings_index[word] = coeffs

    embeddings_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in vocab.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector
        else:
            print(f"No esta {word}")

    return notas, vocab, embeddings_matrix, x_values, y_values

# Method that predicts the next next_words notes.
def prediction(seed_notes, next_words, weights, vocab, max_seq_length, embedding_dim, embeddings_matrix):
    inverse_vocab = {index: token for token, index in vocab.items()}
    predicted_notes = []
    seed_notes_word = notes_to_word(seed_notes)
    encode_seed_notes_words = [vocab[word] for word in seed_notes_word]
    model = create_model(vocab_size=len(vocab), embedding_dim=embedding_dim,
                         embeddings_matrix=embeddings_matrix, max_seq_length=max_seq_length
                         )
    model.load_weights(weights)
    for i in range(next_words):
        input_sequences = generate_n_grams(encode_seed_notes_words, max_len=max_seq_length-1)
        padded_sequences = pad_n_grams(input_sequences)

        predict_result = model.predict(padded_sequences, verbose=0)
        predicted_list = np.argmax(predict_result, axis=-1)
        predicted = predicted_list[-1]
        next_note_word = inverse_vocab[predicted]
        predicted_notes.append(next_note_word)
        print(f"predicted={predicted} ({next_note_word}) : {i+1}/{next_words}")
        encode_seed_notes_words.append(predicted)

    return predicted_notes

# Method to train the model.
def train(model, x_values, y_values):
    savepath = os.path.join("weights", "run_10_weights-1{epoch:02d}-{loss:.4f}.hdf5")
    checkpoint = ModelCheckpoint(
        savepath, monitor='loss',
        verbose=0, save_best_only=True,
        mode='min', save_freq=20,
    )

    callbacks_list = [checkpoint]
    history = model.fit(x_values, y_values,
                        epochs=120, validation_split=0.2,
                        verbose=1, batch_size=32,
                        callbacks=callbacks_list
                        )
    return history

# Method that generates n_notes notes given weights,
# selects randomly a position in the training data and gets 32 notes to start with.
def generate_notes(n_notes, weights, notas=os.path.join('data', 'notas.csv'), max_seq_length=32, embedding_dim=3):

    notas, vocab, embeddings_matrix, _, _ = prepare_data(notas, max_seq_length, embedding_dim)
    random_index = random.randint(0, len(notas)-32)
    seed_notes = notas[random_index:random_index + 32]
    predicted_notes = prediction(seed_notes=seed_notes, next_words=n_notes, weights=weights,
                                 vocab=vocab, max_seq_length=max_seq_length,
                                 embedding_dim=embedding_dim, embeddings_matrix=embeddings_matrix)
    return words_to_notes(predicted_notes)

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    NOTAS = os.path.join('data', 'notas.csv')
    WINDOW_SIZE = 5
    SEED = 42
    BATCH_SIZE = 32
    BUFFER_SIZE = 100
    NUM_NS = 4
    AUTOTUNE = tf.data.AUTOTUNE
    WEIGHTS = os.path.join('weights', 'run_-1_weights-1102-0.2022.hdf5')
    MAX_SEQ_LENGTH = 32
    EMBEDDING_DIM = 3

    notas, vocab, embeddings_matrix, x_values, y_values = prepare_data(NOTAS, MAX_SEQ_LENGTH, EMBEDDING_DIM)

    model = create_model(len(vocab), EMBEDDING_DIM, embeddings_matrix, MAX_SEQ_LENGTH)
    train(model, x_values, y_values)

    random_index = 110
    seed_notes = notas[random_index:random_index+32]

    predicted_notes = prediction(seed_notes=seed_notes, next_words=64, weights=WEIGHTS,
                                 vocab=vocab, max_seq_length=MAX_SEQ_LENGTH,
                                 embedding_dim=EMBEDDING_DIM, embeddings_matrix=embeddings_matrix)

    output_fp = os.path.join("generated_songs", "lstm_output.midi")
