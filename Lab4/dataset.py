import torch
from torch.utils.data import Dataset
from collections import defaultdict
from random import choice
import torchvision


class MNISTMetricDataset(Dataset):
    def __init__(self, root="./mnist/", split='train', remove_classes=None):
        super().__init__()

        assert split in ['train', 'test', 'traineval']

        self.root = root
        self.split = split

        dataset = torchvision.datasets.MNIST(self.root, train='train' in split, download=True)

        self.images, self.targets = dataset.data.float() / 255., dataset.targets
        self.classes = list(range(10))
        self.remove_classes = remove_classes

        if remove_classes is not None:
            [self.classes.remove(x) for x in self.remove_classes]

            indices_to_keep = list(range(len(dataset)))

            for i, target in enumerate(self.targets):
                if target in remove_classes:
                    indices_to_keep.remove(i)

            indices_to_keep = torch.tensor(indices_to_keep)

            self.images = torch.index_select(input=self.images, dim=0, index=indices_to_keep)
            self.targets = torch.index_select(input=self.targets, dim=0, index=indices_to_keep)

        self.target2indices = defaultdict(list)

        for i in range(len(self.images)):
            self.target2indices[self.targets[i].item()] += [i]

    def _sample_negative(self, index):
        return choice(self.target2indices[choice([x for x in self.classes if x != self.targets[index].tolist()])])

    def _sample_positive(self, index):
        return choice(self.target2indices[self.targets[index].tolist()])

    def __getitem__(self, index):
        anchor = self.images[index].unsqueeze(0)
        target_id = self.targets[index].item()

        if self.split in ['traineval', 'val', 'test']:
            return anchor, target_id
        else:
            positive = self._sample_positive(index)
            negative = self._sample_negative(index)

            positive = self.images[positive]
            negative = self.images[negative]

            return anchor, positive.unsqueeze(0), negative.unsqueeze(0), target_id

    def __len__(self):
        return len(self.images)
