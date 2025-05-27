import numpy as np
import random

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.position = 0
        self.alpha = alpha 

    def add(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities, default=1.0)
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(max_priority)
        else: # if the buffer is full, replace the oldest experience
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.priorities[self.position] = max_priority
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = np.array(self.priorities) 
        else:
            priorities = np.array(self.priorities[:len(self.buffer)])

        probabilities = priorities ** self.alpha # compute sampling probabilities
        probabilities /= probabilities.sum() # normalize to get probabilities

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities) # sample indices based on probabilities
        samples = [self.buffer[idx] for idx in indices] # retrieve samples

        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta) # compute importance sampling weights
        weights /= weights.max() # normalize weights
        weights = np.array(weights, dtype=np.float32)

        states, actions, rewards, next_states, dones = map(np.array, zip(*samples))
        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)
