import random
import torch

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = zip(*batch)
        return torch.cat(state), torch.cat(action), torch.cat(reward), torch.cat(next_state)
    def clear(self):
        self.buffer = []
        self.position = 0

    def clear_half(self):
        half_size = len(self.buffer) // 2
        self.buffer = self.buffer[half_size:]
        self.position = len(self.buffer)
    def __len__(self):
        return len(self.buffer)