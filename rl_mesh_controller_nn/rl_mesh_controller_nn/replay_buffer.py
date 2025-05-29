import random
import torch

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (torch.cat(state), torch.cat(action),
                torch.tensor(reward, dtype=torch.float32).view(-1, 1),
                torch.cat(next_state),
                torch.tensor(done, dtype=torch.float32).view(-1, 1))
    def clear(self):
        self.buffer = []
        self.position = 0

    def clear_half(self):
        half_size = len(self.buffer) // 2
        self.buffer = self.buffer[half_size:]
        self.position = len(self.buffer)
    def __len__(self):
        return len(self.buffer)