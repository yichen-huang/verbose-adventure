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