# part 4: Recurrent Neural Network
# RNN => prediction by turning the sequence int vector to get the general meaning
#
# example:
#   Today the weather is gorgeous and I see a beautiful blue ...
#   we would say SKY because of the context BLUE(mainly) and WEATHER
#
# RNN takes part of the previous output or another value(feed forward vallue)
# to use as input for the next iteration
#
# HOWEVER, it's limited to short term memory
# (short term mem = the words/context closest to)
#
# example:
#   I lived in Ireland, so at school they made me learn how to speak ...
#   the answer is GAELIC however the context  IRELAND is further away
#   so in this example shot term memory is limited
#
# That's hy we need a longer mem LONG SHORT-TERM MEMORY LSTM
#
# Part 5: LSTM
#
# in RNN:
#   each time the CONTEXT is passed the futrhest word meaning gets diluted
#
# with LSTM architecture
#   CONTEXT + CELL STATE(CONTEXT that can retain over many iteration)
#       It can be bidirectional
#       (later words can bring meaning to early words and vice-versa)
#
# Part 6: Training on Irish poetry
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np

import matplotlib.pyplot as plt


tokenizer = Tokenizer()

data = open('../dataset/irish-lyrics-eof.txt').read()
corpus = data.lower().split('\n')

# word index
tokenizer.fit_on_texts(corpus)
# +1 is for OUT OF WORD token
total_words = len(tokenizer.word_index) + 1
# in classification you need an OOV now
# but for generating, there won't be any OOV, so the +1 for the padding
# because the padding will count as a word

########################################################## prepare the training set

# n_gram: create a set of sequences only using one line
# example token_list = [1,2,3,4] of len 4 will create 4-1 total set
# [1,2]
# [1,2,3]
# [1,2,3,4]
# you want to train the model to predict the likely next word
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

########################################################## padding

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
# when padded the last element can be used as label
# [0,0,1,2] input([0,0,1]), label(2)
# [0,1,2,3] input([0,1,2]), label(3)
# [1,2,3,4] input([1,2,3]), label(4)

########################################################## labeling

xs = input_sequences[:,:-1]
labels = input_sequences[:,-1]

# used to predict which words in the entire corpus is the most likely word te be next
ys = tf.keras.utils.to_categorical(labels, num_classes = total_words)

########################################################## model + train

model = Sequential()
# max_sequence_len-1 because the last element is used as label
model.add(Embedding(total_words, 240, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(150)))
# output DENSE
model.add(Dense(total_words, activation='softmax'))
adam = Adam(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
history = model.fit(xs, ys, epochs=30, verbose=1)


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.show()

plot_graphs(history, 'accuracy')

########################################################## Test

seed_text = "I've got a bad feeling about this"
next_words = 100

#for _ in range(next_words):
#    token_list = tokenizer.texts_to_sequences([seed_text])[0]
#    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
#    predicted = model.predict_classes(token_list, verbose=0)
#    output_word = ''
#    for word, index in tokenizer.word_index.items():
#        if index == predicted:
#            output_word = word
#            break
#    seed_text += " " + output_word
#
#print(seed_text)

for _ in range(next_words):
	token_list = tokenizer.texts_to_sequences([seed_text])[0]
	token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
	predict_x=model.predict(token_list, verbose=0)
	classes_x=np.argmax(predict_x,axis=1)
	output_word = ""
	for word, index in tokenizer.word_index.items():
		if index == classes_x:
			output_word = word
			break
	seed_text += " " + output_word
print(seed_text)
