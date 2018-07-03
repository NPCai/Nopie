# Nopie

Neural OPen infomation extractor, based on our [Squadie](https://github.com/NPCai/Squadie) dataset. Uses a sequence to sequence model with pre-trained [GloVe embeddings](https://nlp.stanford.edu/projects/glove/) trained with RL.

## Overview

The training procedure alternates supervised learning of our Squadie dataset and reinforcement learning to train the network.

### Supervised Learning

The supervised component comes from standard training of the seq2seq model with attention on the Squadie dataset. 

### Reinfocement Learning

There are numerous possible extractions for a given complex sentence, many of which are not included in Squadie. Using only supervised learning would expectedly lead to high precision and low recall. To improve recall, we alternate supervised training with reinforcement learning. When in RL training mode, a differentiable decoding procedure is used (sampling from a softmax distribution with low temperature at each timestep ~= greedy decoding). We compare the extracted tuple for correctness (e.g. having 3 arguments, noun chunks at the tail, prepositional phrase in the middle).

## Prerequisites

* python 3
* pytorch (>0.4)
* pandas
* spacy
* csv
