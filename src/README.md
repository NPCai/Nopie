### model.py

Contains the encoder and decoder for the recurrent neural network. 

### train.py

This file implements the recurrent neural network object (combining the encoder/decoder from model.py into a seq2seq model) and defines the training code.

### utils.py

Uses the pandas library to extract data from a set of pretrained GloVe word vectors. This file implements key functions that can generate transition between, words/strings, vectors, and a one-hot encoded structure.

### dataLoader.py

Takes the data in ../data/tuples-train.json and adds it to a dataset that can be fed through the neural network for training.

### main.py

Used to generate JSON file with the neural network's produced tuples. It takes in tuple sentence pairs from the Squadie output dataset and runs it through the RNN encoder decoder and the train.py framework.
