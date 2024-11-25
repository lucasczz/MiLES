import numpy as np
from typing import Tuple, List, Optional


def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))  # Stabilize softmax for numerical safety
    return exp_logits / exp_logits.sum()


class ReplayBuffer:
    def __init__(self, recency_temperature: float, max_size: int):
        """
        Initializes an empty buffer with a maximum size.

        Args:
            recency_temperature (float): Temperature parameter to control recency weighting.
            max_size (int): The maximum size of the buffer.
        """
        self.max_size = max_size
        self.rows = np.empty((max_size,), dtype=object)  # Use an object array to store tuples
        self.weights = -np.ones((max_size,)) * np.inf  # Recency weights
        self.recency_factor = 1 / recency_temperature
        self.size = 0
        self.most_recent_idx = -1  # Tracks the index of the most recent row

    def __len__(self):
        return self.size

    def retrieve(
        self, n_current: int, row: Tuple
    ) -> Tuple[Optional[List], Optional[List]]:
        """
        Adds a tuple to the buffer. If the buffer exceeds max_size, replaces a random sample.
        Ensures the most recent row is included in the `n_current` samples.
        Returns `retrieved` and `remaining` tuples in a transposed fashion.

        Args:
            n_current (int): Number of tuples to retrieve.
            row (Tuple): Tuple to add to the buffer.

        Returns:
            Tuple[Optional[List], Optional[List]]: A tuple containing two lists:
                - `retrieved`: List of tuples (transposed if possible).
                - `remaining`: List of tuples (transposed if possible).
        """
        # Add the new row to the buffer
        if self.size < self.max_size:
            self.rows[self.size] = row
            self.weights[self.size] = self.size * self.recency_factor
            self.most_recent_idx = self.size
            self.size += 1
        else:
            # Replace a random sample when max_size is reached
            idx = np.random.randint(self.size)
            self.rows[idx] = row
            self.weights -= self.recency_factor  # Adjust weights for recency
            self.weights[idx] = self.max_size * self.recency_factor
            self.most_recent_idx = idx

        if self.size == 0:
            return None, None

        # Always include the most recent row
        n_random = max(0, n_current - 1)
        other_indices = np.delete(np.arange(self.size), self.most_recent_idx)
        
        # Sample the remaining rows
        sampled_indices = (
            np.random.choice(
                other_indices,
                size=min(n_random, len(other_indices)),
                replace=False,
                p=softmax(self.weights[other_indices]),
            )
            if n_random > 0
            else []
        )
        
        # Combine the most recent row with sampled rows
        sampled_indices = np.append(sampled_indices, self.most_recent_idx).astype(np.int32)
        retrieved = self.rows[sampled_indices]

        # Determine remaining rows
        remaining = np.delete(self.rows[:self.size], sampled_indices)

        # Transpose `retrieved` and `remaining` if they exist
        try:
            retrieved_transposed = list(zip(*retrieved))
        except TypeError:
            retrieved_transposed = None

        if len(remaining) > 0:
            remaining_transposed = list(zip(*remaining))
        else:
            remaining_transposed = (None, None, None, None)

        return retrieved_transposed, remaining_transposed
