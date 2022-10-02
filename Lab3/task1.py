from dataclasses import dataclass, astuple

import torch
import pandas as pd
from torch.utils.data import Dataset


@dataclass
class Instance:
    text: str
    label: str

    def __iter__(self):
        return iter(astuple(self))


class NLPDataset(Dataset):
    def __init__(self, path, text_vocabulary, label_vocabulary):
        df = pd.read_csv(path, header=None)
        self.instances = [([token for token in row[0].split()], row[1].strip()) for index, row in df.iterrows()]

        self.text_vocabulary = text_vocabulary
        self.label_vocabulary = label_vocabulary

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        numericalized_text = torch.tensor(self.text_vocabulary.encode(self.instances[idx][0]))
        numericalized_label = torch.tensor(self.label_vocabulary.encode(self.instances[idx][1]))

        return numericalized_text, numericalized_label


class Vocabulary:
    def __init__(self, frequencies, max_size=-1, min_freq=0, special_symbols=None):
        self.max_size, self.min_freq = max_size, min_freq
        self.itos, self.stoi = dict(), dict()

        frequencies = {word: frequency for word, frequency in frequencies.items() if frequency >= min_freq}
        frequencies = {k: v for k, v in sorted(frequencies.items(), key=lambda item: item[1], reverse=True)}
        frequencies = list(frequencies)[:max_size] if max_size != -1 else frequencies

        if special_symbols is not None:
            for i, symbol in enumerate(special_symbols):
                if 0 <= max_size <= len(self.itos):
                    break

                self.itos[i] = symbol
                self.stoi[symbol] = i

        for i, symbol in enumerate(frequencies, start=len(self.itos)):
            if 0 <= max_size <= len(self.itos):
                break

            self.itos[i] = symbol
            self.stoi[symbol] = i

    def encode(self, tokens, fallback_token="<UNK>"):
        if isinstance(tokens, list):
            return [self.stoi.get(token, self.stoi[fallback_token]) for token in tokens]

        return self.stoi.get(tokens, fallback_token)

    def __len__(self):
        return len(self.itos)
