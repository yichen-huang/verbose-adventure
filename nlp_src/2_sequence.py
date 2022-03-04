import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
# to handle sentences of different lengths we use padding
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]

# num_words: take the X most present words
# oov_token: when sequencing, if an unknown word appears it will take "<OOV>" (out of value)
tokenizer = Tokenizer(num_words= 100, oov_token = "<OOV>")
tokenizer.fit_on_texts(sentences)
# list containing index of words
word_index = tokenizer.word_index
print ('word_index :', word_index)


# turn the sentences into list of index
sequences = tokenizer.texts_to_sequences(sentences)
print('sequences : ', sequences)

# now every sequences have the same lengths
# padded with Zeros
# padding = "post": put the padding after the sentences
# padding = "pre": put the padding before the sentences
# maxlen: desired length of the sequences
# truncating = "post": if the sentences is above maxlen the sentences are truncated at the end
# truncating = "pre": if the sentences is above maxlen the sentences are truncated at the beginning
padded = pad_sequences(sequences, maxlen = 10)
print('padded sequences :\n', padded)

test_data = [
    'I really love my dog', # really is unknown
    'my dog loves my manatee' # loves, manatee are unkown
]

test_seq = tokenizer.texts_to_sequences(test_data)
print('test_sequences with unknown words : ',test_seq)