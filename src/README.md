### model.py

Contains the encoder and decoder for the recurrent neural network.

### train.py

This file implements the recurrent neural network object and also trains the network. Using a cross entropy loss function, and an adam optimization algorithm, it encodes the tuple data and then calculates loss based on the decoded output.

### utils.py

Uses the pandas library to extract data from a set of pretrained GloVe word vectors. This file implements four key functions that can generate vectors from words or strings and also generate words or strings from vectors.
This is important

### dataLoader.py

Takes the data in ../data/tuples-train.json and adds it to a dataset that can be fed through the neural network for training.

### main.py

Used to generate 
