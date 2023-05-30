import csv
from typing import List, Dict
import io
import tqdm
import os
import glob
from music21 import note, chord, midi
import tensorflow as tf
from keras.utils.vis_utils import plot_model
from word2vec import Word2Vec

# Global variables
NOTAS = 'data/notas.csv'
WINDOW_SIZE = 5
SEED = 42
BATCH_SIZE = 32
BUFFER_SIZE = 100
NUM_NS = 4
AUTOTUNE = tf.data.AUTOTUNE


# NOT IN USE. Method that obtains all the notes from the data/midi_songs/ directory if data/notas.csv does not exist.
def get_notes():
    filename = os.path.join('data', 'notas.csv')
    notes = []
    if os.path.exists(filename):
        print("notes already have been processed")
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';', quoting=csv.QUOTE_NONNUMERIC)
            for row in csv_reader:
                notes.append(row)
    else:
        for file in glob.glob("data/midi_songs/*.mid"):
            print("Parsing %s" % file)
            midi_file = midi.MidiFile()
            midi_file.open(file)
            midi_file.read()
            midi_stream = midi.translate.midiFileToStream(midi_file)
            for element in midi_stream.flat:
                if isinstance(element, note.Note):
                    notes.append([element.pitch.midi, 0, 0, 0, 0, 0, 0, 0])
                elif isinstance(element, chord.Chord):
                    notes_chord = [chordnote.pitch.midi for chordnote in element]
                    notes_chord.extend([0] * (8 - len(element)))
                    notes.append(notes_chord)
                # elif isinstance(element, note.Rest):
                #     notes.append([0] * 8)
            with open(filename, mode='w', newline='') as fil:
                writer = csv.writer(fil)
                for row in notes:
                    writer.writerow(row)
    return notes

# Method that reads the notes from a csv.
def read_notes_csv(fname: str) -> List:
    rows = list()
    with open(fname) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';', quoting=csv.QUOTE_NONNUMERIC)
        for row in csv_reader:
            rows.append(row)
    return rows


# Method that transforms the notes (list of ints) into words (string).
def notes_to_word(notes: List) -> List:
    notes_words = list()
    for note in notes:
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
def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
    # Elements of each training example are appended to these lists.
    targets, contexts, labels = [], [], []

    # Build the sampling table for `vocab_size` tokens.
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

    # Iterate over all sequences (sentences) in the dataset.
    for sequence in tqdm.tqdm(sequences):
        # Generate positive skip-gram pairs for a sequence (sentence).
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            sequence,
            vocabulary_size=vocab_size,
            sampling_table=sampling_table,
            window_size=window_size,
            negative_samples=0)

        # Iterate over each positive n-gram pair to produce training examples
        # with a positive context word and negative samples.
        for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(
                tf.constant([context_word], dtype="int64"), 1)
            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class,
                num_true=1,
                num_sampled=num_ns,
                unique=True,
                range_max=vocab_size,
                seed=seed,
                name="negative_sampling")

            # Build context and label vectors (for one target word)
            context = tf.concat([tf.squeeze(context_class, 1), negative_sampling_candidates], 0)
            label = tf.constant([1] + [0] * num_ns, dtype="int64")

            # Append each element from the training example to global lists.
            targets.append(target_word)
            contexts.append(context)
            labels.append(label)

    return targets, contexts, labels


if __name__ == '__main__':

    get_notes()

    notas = read_notes_csv(NOTAS)
    print(len(notas))
    print(notas[:5])
    notes_word = notes_to_word(notas)
    assert len(notes_word) > 0
    # ['048055063070074000000000', ..., '048055063070074000000000']
    print(notes_word[:5])
    vocab = create_vocab(notes_word)
    inverse_vocab = {index: token for token, index in vocab.items()}
    vocab_size = len(vocab)
    window_size = WINDOW_SIZE

    encode_notes_words = [vocab[word] for word in notes_word]
    # [1, 2, 3, 4, 1]
    print(encode_notes_words[:5])

    sequences = generate_sequences(encode_notes_words, sequence_size=32, ovelap_size=1)

    targets, contexts, labels = generate_training_data(
        sequences=sequences,
        window_size=WINDOW_SIZE,
        num_ns=NUM_NS,
        vocab_size=vocab_size,
        seed=SEED)

    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    embedding_dim = 3
    word2vec = Word2Vec(vocab_size, embedding_dim, num_ns=NUM_NS)
    word2vec.compile(optimizer='adam',
                     loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])

    word2vec.fit(dataset, epochs=5 ) #callbacks=[tensorboard_callback])
    # plot_model(word2vec, to_file='encoder_model_plot.png', show_shapes=True, show_layer_names=True)

    # Save data
    weights = word2vec.get_layer('w2v_embedding').get_weights()[0]

    out_v = io.open('embedding_vectors.csv', 'w', encoding='utf-8')

    for index, word in enumerate(vocab):
        if index == 0:
            continue  # skip 0, it's padding.
        vec = weights[index]
        out_v.write(word + ' ' + ' '.join([str(x) for x in vec]) + "\n")

    out_v.close()
