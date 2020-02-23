# entry point of the program
# variables
num_of_train_data = 1000
vocab_size = 100
embedding_dim = 100
max_length = 100
trunc_type = 'post'
oov_tok = "<OOV>"
training_size = 200
epoch = 30

import pandas as pd
import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# DO NOT CHANGE THIS
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

# DO NOT CHANGE THIS

# FUNCTION
def remove_speaker(_input):
    # only show relevant candidate, and remove speaker and moderator
    candidate = {
        'Joe Biden': 1,
        'Bernie Sanders': 2,
        'Amy Klobachar': 3,
        'Tom Steyer': 4,
        'Elizabeth Warren': 5,
        'Pete Buttigieg': 6,
    }

    _input['speaker'] = _input['speaker'].map(candidate)

    _input = _input.dropna()
    return _input

def clean_speech(_input):
    stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
                 "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did",
                 "do",
                 "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
                 "having",
                 "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his",
                 "how",
                 "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself",
                 "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other",
                 "ought",
                 "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should",
                 "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
                 "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
                 "through",
                 "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were",
                 "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom",
                 "why",
                 "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
                 "yourselves"];

    sentences = [];

    for a in _input['speech']:
        for b in stopwords:
            token = " " + b + " "
            a = a.replace(token, " ")
        sentences.append(a)

    _input['speech'] = sentences
    return _input


def plot_graphs(_history, string):
    plt.plot(_history.history[string])
    plt.plot(_history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

# END FUNCTION


# load the data
data = pd.read_csv('data/debate_transcripts.csv')

clean_feature = remove_speaker(data)
clean_feature = clean_speech(data)

print(clean_feature)

#split train and test
train_data = clean_feature[:num_of_train_data]
test_data = clean_feature[num_of_train_data:]

train_speech = train_data['speech']
test_speech = test_data['speech']

train_speaker = train_data['speaker']
test_speaker = test_data['speaker']

# clean up speech
tokenizer = Tokenizer(oov_token=oov_tok, num_words=vocab_size)
tokenizer.fit_on_texts(train_speech)
word_index = tokenizer.word_index
train_sequences = tokenizer.texts_to_sequences(train_speech)
train_padded = pad_sequences(train_sequences, maxlen=max_length, truncating=trunc_type)


testing_sequence = tokenizer.texts_to_sequences(test_speech)
testing_padded = pad_sequences(testing_sequence, maxlen=max_length, truncating=trunc_type)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(train_padded, train_speaker, epochs=epoch,
                    validation_data=(testing_padded, test_speaker))

plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

e = model.layers[0]
weights = e.get_weights()[0]

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
'''
import io

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
  word = reverse_word_index[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()
'''