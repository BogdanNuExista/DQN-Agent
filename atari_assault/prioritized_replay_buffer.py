# prioritized_replay_buffer.py
import numpy as np

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.alpha = alpha
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        state = (state * 255).astype(np.uint8)
        next_state = (next_state * 255).astype(np.uint8)
        
        max_prio = np.max(self.priorities) if self.size > 0 else 1.0
        transition = (state, action, reward, next_state, done)
        
        if self.size < self.capacity:
            self.buffer.append(transition)
            self.priorities[self.size] = max_prio
            self.size += 1
        else:
            self.buffer[self.position] = transition
            self.priorities[self.position] = max_prio
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(self.size, batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        states = np.stack([s[0].astype(np.float32)/255 for s in samples])
        next_states = np.stack([s[3].astype(np.float32)/255 for s in samples])

        actions = np.array([s[1] for s in samples])
        rewards = np.array([s[2] for s in samples], dtype=np.float32)
        dones = np.array([s[4] for s in samples], dtype=np.bool_)

        total = self.size
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = weights.astype(np.float32)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, priorities):
        priorities = np.abs(priorities) + 1e-6
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return self.size