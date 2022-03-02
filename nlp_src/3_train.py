####################################################preprocessing
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


####################################################json to python list
import json

# json.load() doesn't work for some obscure reasons
#datastore = []
#for line in open('sarcasm.json', 'r'):
#    datastore.append(json.loads(line))
datastore = [json.loads(line) for line in open('../dataset/sarcasm.json', 'r')]

sentences = []
labels = []
urls = []
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

####################################################prep sets

vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type='post'
padding_type='post'
oov_tok = '<OOV>'
training_size = 20000

training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]

training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

####################################################
#preprocessing + splitting the sentence into training set and test set

tokenizer = Tokenizer(num_words= vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length,
                                padding= padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length,
                                padding= padding_type, truncating=trunc_type)

####################################################embedding
# embedding: give a value to a notion
# example on a graph: notions of good, ok, bad
#                       y
#
#                      ok [0,1]
#                       |
#                       |
#                       |
#                       |
#                       |
# bad [-1,0]            |                    good[1,0]
# ______________________________________________X

#################################################### Neural Network

training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation= 'relu'),
    tf.keras.layers.Dense(1, activation= 'sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#model.summary()

#################################################### training

num_epochs = 30
history = model.fit(training_padded, training_labels, epochs=num_epochs,
                    validation_data=(testing_padded, testing_labels),
                    verbose=2)
# Apparently the model is not saved

#################################################### testing with new sentences

sentence = [
    "granny starting to fear spiders in the garden might be real",
    "game of thrones season finale showing this sunday night"
]

sequences = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sequences, maxlen=max_length,
                        padding=padding_type,
                        truncating=trunc_type)
print(model.predict(padded))