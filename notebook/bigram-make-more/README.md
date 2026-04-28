# Makemore

This is a character level language model. We are trying to predict the next character in a sequence of characters.

*names.txt* - dataset with 32,000 names.

Current language model neural nets implemented:

* Bigram (one character simply pridicts a next one with a lookup table of counts)
* Bag of Words (considers a window of characters before the current one)
* MLP (Multi Layer Perceptron)
* RNN (Recurrent Neural Network)
* GRU (Gated Recurrent Unit)
* Transformer (with self-attention)