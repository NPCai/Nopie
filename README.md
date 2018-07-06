# Nopie

Neural OPen infomation extractor, based on our [Squadie](https://github.com/NPCai/Squadie) dataset. Uses a sequence to sequence model with pre-trained [GloVe embeddings](https://nlp.stanford.edu/projects/glove/).

## Overview

The model consists of a standard seq2seq model with attention, but with two extra inputs for NER chunks. The model is trained to produce the relation between the chunks. The two chunks are used as input to a "directed" attention model.

## Prerequisites

* python 3
* pytorch (>0.4)
* pandas
* spacy
* csv
