import numpy as np
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        """
        Initialize a Replay Buffer.
        
        Args:
            buffer_size (int): Maximum size of the buffer
            batch_size (int): Size of training batches to sample
        """
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
    
    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self):
        """
        Sample a batch of experiences from the buffer.
        
        Returns:
            tuple: Batch of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, min(len(self.buffer), self.batch_size))
        
        states = np.array([experience[0] for experience in batch])
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch])
        next_states = np.array([experience[3] for experience in batch])
        dones = np.array([experience[4] for experience in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)