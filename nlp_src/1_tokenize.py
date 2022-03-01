import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!'
]

# num_words: take the X most present words
tokenizer = Tokenizer(num_words= 100)
tokenizer.fit_on_texts(sentences)
# list containing index of words
word_index = tokenizer.word_index

print (word_index)