#!/bin/sh

SQUAD1="squad/squad1"
mkdir -p $SQUAD1
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O ${SQUAD1}/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O ${SQUAD1}/dev-v1.1.json

SQUAD2="squad/squad2"
mkdir -p $SQUAD2
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json -O ${SQUAD2}/train-v2.0.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O ${SQUAD2}/dev-v2.0.json
