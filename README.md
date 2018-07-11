# Nopie

Neural OPen infomation extractor, based on our [Squadie](https://github.com/NPCai/Squadie) dataset and bootstrapped from the [Graphene OpenIE system](https://github.com/Lambda-3/Graphene).

## Overview

*Model*: Attention is All You Need (Transformer)

*General Training procedure*: Initially train the transformer to mimic the Graphene OpenIE system (i.e. “bootstrapping”) until convergence. Then alternate with REINFORCE on our Squadie and News QA IE datasets to improve. Rewards are given for (1) NER’s being identified (as provided by an external NER system) (2) tuples being identified, (3) correct causal and temporal tags. (See: MIXER training in https://arxiv.org/pdf/1511.06732.pdf)

## Milestones

- [x] Make Squadie Dataset
- [x] Make News QA Dataset
- [x] Create a seq2seq with attention baseline on the datasets using OpenNMT (just for fun)
- [x] Construct a database using the Graphene parser
- [ ] Train a transformer model on the graphene database
- [ ] Use MIXER training to improve on the model

## Prerequisites

* python 3
* pytorch (>0.4)
* pandas
* spacy
* csv
