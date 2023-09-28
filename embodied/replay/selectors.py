from collections import deque
import torch
import numpy as np


class Fifo:
    def __init__(self):
        self.queue = deque()

    def __call__(self):
        return self.queue[0]

    def __setitem__(self, key, steps):
        self.queue.append(key)

    def __delitem__(self, key):
        if self.queue[0] == key:
            self.queue.popleft()
        else:
            # TODO: This branch is unused but very slow.
            self.queue.remove(key)


class Uniform:
    def __init__(self, seed=0):
        self.indices = {}
        self.keys = []
        self.rng = np.random.default_rng(seed)

    def __call__(self):
        index = self.rng.integers(0, len(self.keys)).item()
        return self.keys[index]

    def __setitem__(self, key, steps):
        self.indices[key] = len(self.keys)
        self.keys.append(key)

    def __delitem__(self, key):
        index = self.indices.pop(key)
        last = self.keys.pop()
        if index != len(self.keys):
            self.keys[index] = last
            self.indices[last] = index


class TimeBalanced:
    def __init__(self, seed=0, bias_factor=1.5):
        self.indices = {}
        self.keys = []
        self.rng = np.random.default_rng(seed)
        self.bias_factor = bias_factor
        self.distribution = torch.distributions.Beta(self.bias_factor, 1)

    def __call__(self):
        sample = self.distribution.sample()
        index = int(sample * len(self.keys))
        return self.keys[index]

    def __setitem__(self, key, steps):
        self.indices[key] = len(self.keys)
        self.keys.append(key)

    def __delitem__(self, key):
        index = self.indices.pop(key)
        last = self.keys.pop()
        if index != len(self.keys):
            self.keys[index] = last
            self.indices[last] = index


class TimeBalancedNaive:
    def __init__(self, seed=0, bias_factor=20):
        self.indices = {}
        self.keys = []
        self.rng = np.random.default_rng(seed)
        self.bias_factor = bias_factor
        # self.distribution = torch.distributions.Beta(self.bias_factor, 1)
        self.key_counts = []
        self.sample_count = 1
        self.is_counter = 0

    def __call__(self):
        if self.sample_count <= 1:
            index = 0
        else:
            probs = -1 * (torch.tensor(self.key_counts) / self.bias_factor)
            probs = torch.softmax(probs, dim=0)
            index = torch.distributions.Categorical(probs=probs).sample().item()
        self.key_counts[index] += 1
        self.sample_count += 1
        return self.keys[index]

    def __setitem__(self, key, steps):
        self.indices[key] = len(self.keys)
        self.keys.append(key)
        self.key_counts.append(0)

    def __delitem__(self, key):
        index = self.indices.pop(key)
        last = self.keys.pop()
        if index != len(self.keys):
            self.keys[index] = last
            self.indices[last] = index


class EfficientTimeBalanced:
    def __init__(self, seed=0, length=1e6, temperature=1.0):
        self.indices = {}
        self.keys = []
        self.rng = np.random.default_rng(seed)
        self._max_len = length
        n = length + 2
        self._b = (n - 2) * np.log(n) + np.log(n - 1) + n * (1 - np.log(n - 1)) - 2
        assert 0 <= temperature <= 1
        self._temperature = temperature
        self.key_counts = []
        self.sample_count = 1
        self._is_counter = 0

    def __call__(self):
        sample = self.rng.uniform(low=2, high=len(self.keys) + 2)
        adjusted_sample = self.cdf(sample)
        index = int(
            adjusted_sample * self._temperature + (1 - self._temperature) * (sample - 2)
        )
        self.key_counts[index] += 1
        self.sample_count += 1
        return self.keys[index]

    def __setitem__(self, key, steps):
        self.indices[key] = len(self.keys)
        self.keys.append(key)
        self.key_counts.append(0)

    def __delitem__(self, key):
        index = self.indices.pop(key)
        last = self.keys.pop()
        if index != len(self.keys):
            self.keys[index] = last
            self.indices[last] = index

    def cdf(self, x):
        # this function is only defined for 2 >= x < n + 2
        n = len(self.keys) + 2

        def integral(y):
            a = y * (np.log(y) - np.log(n) - 1)
            b = 2 * np.log(n) - n - 2 * np.log(2) + 2
            return a / b

        true_uniform_sample = integral(x) - integral(2)
        sampled_index = min(true_uniform_sample * self._max_len, len(self.keys) - 1)
        return sampled_index
