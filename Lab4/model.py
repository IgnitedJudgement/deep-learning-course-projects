import torch
from torch import linalg as LA
import torch.nn as nn


class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, bias=True):
        super(_BNReluConv, self).__init__()

        self.add_module("Batch normalization", nn.BatchNorm2d(num_maps_in))
        self.add_module("ReLU", nn.ReLU())
        self.add_module("Convolution", nn.Conv2d(num_maps_in, num_maps_out, k, bias=bias))


class SimpleMetricEmbedding(nn.Module):
    def __init__(self, input_channels, emb_size=32):
        super().__init__()

        self.reductions = {
            "none": lambda x: x,
            "mean": torch.mean,
            "sum": torch.sum
        }

        self.emb_size = emb_size

        self.block1 = _BNReluConv(input_channels, emb_size)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.block2 = _BNReluConv(emb_size, emb_size)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.block3 = _BNReluConv(emb_size, emb_size)
        self.avg_pool = nn.AvgPool2d(kernel_size=2)

    def get_features(self, img):
        result = self.block1(img)
        result = self.max_pool1(result)
        result = self.block2(result)
        result = self.max_pool2(result)
        result = self.block3(result)
        result = self.avg_pool(result)
        result = result.squeeze()
        result = result.view(-1, self.emb_size)

        return result

    def loss(self, anchor, positive, negative, margin=1, p=2, reduction="mean"):
        assert reduction in ["none", "mean", "sum"]

        a_x = self.get_features(anchor)
        p_x = self.get_features(positive)
        n_x = self.get_features(negative)

        p_distance = LA.norm(a_x - p_x, dim=1, ord=p)
        n_distance = LA.norm(a_x - n_x, dim=1, ord=p)

        loss = torch.maximum(p_distance - n_distance + margin, torch.tensor(0))

        return self.reductions[reduction](loss)


class IdentityModel(nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()

    def get_features(self, img):
        return img.view(img.shape[0], -1)
