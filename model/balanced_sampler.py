from torch.utils.data import Sampler
import random


class BalancedSampler(Sampler):
    def __init__(self, positives, negatives, batch_size):
        self.positives = positives
        self.negatives = negatives
        self.batch_size = batch_size
        self.half_batch = batch_size // 2

    def __iter__(self):
        # Shuffle positives and negatives separately
        pos_indices = self.positives.copy()
        neg_indices = self.negatives.copy()
        random.shuffle(pos_indices)
        random.shuffle(neg_indices)

        # Yield batches with half positives and half negatives
        min_len = min(len(pos_indices), len(neg_indices))
        for i in range(0, min_len, self.half_batch):
            batch = pos_indices[i:i+self.half_batch] + \
                neg_indices[i:i+self.half_batch]
            random.shuffle(batch)
            for idx in batch:
                yield idx

    def __len__(self):
        return min(len(self.positives), len(self.negatives)) * 2
