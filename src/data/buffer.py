from typing import Tuple, List
import random
from collections import Counter, deque
from itertools import chain

import torch


class BaseBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity

    def sample(self, batch_size: int) -> List[Tuple]:
        sample = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        # Check if samples have 5 elements (with ll_target) or 4 (original)
        if len(sample[0]) == 5:
            xl, tl, lll, ul, ll_target = list(zip(*sample))
            x = list(chain(*xl))
            t = list(chain(*tl))
            ll = list(chain(*lll))
            u = torch.concat(ul)
            ll_target = torch.cat(ll_target, dim=0)
            return x, t, ll, u, ll_target
        else:
            xl, tl, lll, ul = list(zip(*sample))
            x = list(chain(*xl))
            t = list(chain(*tl))
            ll = list(chain(*lll))
            u = torch.concat(ul)
            return x, t, ll, u

    def __len__(self):
        return len(self.buffer)


class RecentBuffer(BaseBuffer):
    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.buffer = deque(maxlen=capacity)

    def add(
        self, sample: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        self.buffer.append(sample)


class RandomBuffer(BaseBuffer):
    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.buffer: List[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []

    def add(
        self, sample: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            idx = random.randint(0, self.capacity - 1)
            self.buffer[idx] = sample


class BalancedBuffer(BaseBuffer):
    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.buffer: List[Tuple] = []
        self.class_counter = Counter()
        self.labels = torch.empty(0, dtype=torch.long)  # Store labels for fast indexing

    def add(
        self, sample: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        xc, tc, llc, uc = sample
        label = uc.item()
        self.class_counter[label] += 1

        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
            self.labels = torch.cat(
                [self.labels, torch.tensor([label], dtype=torch.long)]
            )
        else:
            # Replace a random sample of the most frequent class
            most_common_class = self.class_counter.most_common(1)[0][0]
            indices = (self.labels == most_common_class).nonzero(as_tuple=True)[0]
            if len(indices) > 0:
                idx = indices[torch.randint(len(indices), (1,)).item()].item()
                self.buffer[idx] = sample
                self.labels[idx] = label
