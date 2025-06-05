import torch
from torch.utils.data import Sampler
import random


class PositiveOversampleSampler(Sampler):
    def __init__(self, labels, num_samples):
        self.labels = labels
        self.num_samples = num_samples

        self.pos_indices = [i for i, label in enumerate(labels) if label == 1]
        self.neg_indices = [i for i, label in enumerate(labels) if label == 0]

    def __iter__(self):
        sampled_indices = []
        num_pos = int(self.num_samples * 0.5)  # 50% positives
        num_neg = self.num_samples - num_pos

        pos_sampled = random.choices(self.pos_indices, k=num_pos)
        neg_sampled = random.choices(self.neg_indices, k=num_neg)

        sampled_indices = pos_sampled + neg_sampled
        random.shuffle(sampled_indices)
        return iter(sampled_indices)

    def __len__(self):
        return self.num_samples
