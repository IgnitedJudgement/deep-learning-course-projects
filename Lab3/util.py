import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from Lab3.task1 import Vocabulary, NLPDataset


def generate_embedding_matrix(vocabulary, padding_idx=0, freeze=False, path=False, dimensionality=300):
    if path is not False:
        word_vector_dict = get_word_vector_dict(path)
        dimensionality = len(list(word_vector_dict.values())[0])

        embeddings = np.random.rand(len(vocabulary), dimensionality)

        for i, word in enumerate(vocabulary.stoi):
            if word == "<PAD>":
                embeddings[i] = np.zeros(dimensionality)
            elif word in word_vector_dict:
                embeddings[i] = word_vector_dict[word]
    else:
        embeddings = np.random.rand(len(vocabulary), dimensionality)

    return nn.Embedding.from_pretrained(embeddings=torch.tensor(embeddings, dtype=torch.float32),
                                        padding_idx=padding_idx, freeze=freeze)


def get_word_vector_dict(path):
    word_vector_dict = dict()

    with open(path) as file:
        for line in file.readlines():
            elements = line.split()
            word_vector_dict[elements[0]] = np.array([float(x) for x in elements[1:]])

    return word_vector_dict


def get_frequencies(path):
    df = pd.read_csv(path, header=None)

    text_df = df[0].str.split(expand=True).stack().value_counts()
    label_df = df[1].str.split(expand=True).stack().value_counts()

    text_frequencies = {word: frequency for word, frequency in text_df.iteritems()}
    label_frequencies = {word: frequency for word, frequency in label_df.iteritems()}

    return text_frequencies, label_frequencies


def pad_collate_fn(batch, pad_index=0):
    given_texts, given_labels = zip(*batch)

    texts = pad_sequence(given_texts, batch_first=True, padding_value=pad_index)
    labels = torch.tensor(given_labels)
    lengths = torch.tensor([len(text) for text in given_texts])

    return texts, labels, lengths


def get_dataloaders(paths, special_symbols, batch_sizes=None):
    # Frequencies
    text_frequencies, label_frequencies = get_frequencies(paths["train"])

    # Vocabularies
    text_vocabulary = Vocabulary(text_frequencies, special_symbols=special_symbols)
    label_vocabulary = Vocabulary(label_frequencies)

    # Datasets
    train = NLPDataset(paths["train"], text_vocabulary, label_vocabulary)
    validation = NLPDataset(paths["validation"], text_vocabulary, label_vocabulary)
    test = NLPDataset(paths["test"], text_vocabulary, label_vocabulary)

    # Dataloaders
    train_dataloader = DataLoader(train, batch_size=batch_sizes["train"], shuffle=True, collate_fn=pad_collate_fn)
    validation_dataloader = DataLoader(validation, batch_size=batch_sizes["validate"], collate_fn=pad_collate_fn)
    test_dataloader = DataLoader(test, batch_size=batch_sizes["test"], collate_fn=pad_collate_fn)

    return train_dataloader, validation_dataloader, test_dataloader
