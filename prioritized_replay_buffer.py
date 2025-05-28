import numpy as np

# buffer that stores transitions with priorities for sampling

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity # maximum number of transitions (experiences) to store
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0 # current position
        self.alpha = alpha # prioritization exponent (1=full prioritization, 0=uniform sampling)
        self.size = 0 # current n.o. exp

    def add(self, state, action, reward, next_state, done):
        # Compress states to save memory (float32 [0,1] -> uint8 [0,255])
        state = (state * 255).astype(np.uint8)
        next_state = (next_state * 255).astype(np.uint8)
        
        # get max priority for the new experience
        max_prio = np.max(self.priorities) if self.size > 0 else 1.0
        transition = (state, action, reward, next_state, done)
        
        # add new experience or replace the oldest one if full
        if self.size < self.capacity:
            self.buffer.append(transition)
            self.priorities[self.size] = max_prio
            self.size += 1
        else:
            self.buffer[self.position] = transition
            self.priorities[self.position] = max_prio
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        # Get priorities and compute sampling probabilities
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        # Sample experiences based on priorities
        indices = np.random.choice(self.size, batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        # Decompress states and organize data
        states = np.stack([s[0].astype(np.float32)/255 for s in samples])
        next_states = np.stack([s[3].astype(np.float32)/255 for s in samples])

        actions = np.array([s[1] for s in samples])
        rewards = np.array([s[2] for s in samples], dtype=np.float32)
        dones = np.array([s[4] for s in samples], dtype=np.bool_)

        # Compute importance sampling weights
        total = self.size
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max() # Normalize weights
        weights = weights.astype(np.float32)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, priorities):
        # Update priorities with new TD errors, larger errors mean higher priority
        priorities = np.abs(priorities) + 1e-6 # ensure non zero prio
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return self.size