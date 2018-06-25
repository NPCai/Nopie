### model.py

Contains the encoder and decoder for the recurrent neural network. While it may seem that the model is lacking in depth, we felt that a minimal implementation would be more than enough for this project.

### train.py

This file implements the recurrent neural network object and also trains the network. Using a cross entropy loss function, and an adam optimization algorithm, it encodes the tuple data and then calculates loss based on the decoded output. The evaluation function will use the calculated probability to form output tuples.

### utils.py

Uses the panda library to extract data from a set of pretrained GloVe word vectors. This file implements key functions that can generate transition between, words/strings, vectors, and a one-hot encoded structure.

### dataLoader.py

Takes the data in ../data/tuples-train.json and adds it to a dataset that can be fed through the neural network for training.

### main.py

Used to generate JSON file with the neural network's produced tuples. It takes in tuple sentence pairs from the Squadie output dataset and runs it through the RNN encoder decoder and the train.py framework.
