
import numpy as np
from tinygrad.tensor import Tensor
import tinygrad.optim as optim
from qnetwork import QNetwork
from replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, state_size, action_size, 
                 buffer_size=10000, 
                 batch_size=64, 
                 gamma=0.99, 
                 lr=1e-3, 
                 update_every=4):
        """
        Initialize a DQN Agent.
        
        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            buffer_size (int): Replay buffer size
            batch_size (int): Minibatch size
            gamma (float): Discount factor
            lr (float): Learning rate
            update_every (int): How often to update the network
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_every = update_every
        self.t_step = 0
        
        # Q-Networks - current and target
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self._update_target()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)
        
        # Optimizer
        self.optimizer = optim.SGD(self.q_network.parameters, lr=lr)
    
    def _update_target(self):
        """Copy weights from Q-network to target network"""
        for q_param, target_param in zip(self.q_network.parameters, self.target_network.parameters):
            target_param.assign(q_param)
    
    def step(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer and learn if it's time.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Store experience in replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        # Learn every update_every time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.replay_buffer) > self.batch_size:
            self._learn()
    
    def act(self, state, epsilon=0.0):
        """
        Get action according to epsilon-greedy policy.
        
        Args:
            state: Current state
            epsilon (float): Exploration rate
            
        Returns:
            int: Selected action
        """
        return self.q_network.get_action(state, epsilon)
    
    def _learn(self):
        """Update Q-Network weights from batch of experiences"""
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
        
        # Convert to tinygrad tensors
        states_t = Tensor(states)
        actions_t = Tensor(actions.reshape(-1, 1))
        rewards_t = Tensor(rewards.reshape(-1, 1))
        next_states_t = Tensor(next_states)
        dones_t = Tensor(dones.reshape(-1, 1).astype(np.float32))
        
        # Get Q-values for current states
        q_values = self.q_network.forward(states_t)
        
        # Get Q-values for next states from target network
        with Tensor.no_grad():
            next_q_values = self.target_network.forward(next_states_t)
            max_next_q = next_q_values.max(axis=1).reshape(-1, 1)
            target_q_values = rewards_t + (self.gamma * max_next_q * (1 - dones_t))
        
        # Get Q-values for actions taken
        q_values_for_actions = Tensor.zeros((q_values.shape[0], 1))
        for i in range(q_values.shape[0]):
            q_values_for_actions.data[i, 0] = q_values.data[i, int(actions_t.data[i, 0])]
        
        # Calculate loss and backpropagate
        loss = ((q_values_for_actions - target_q_values) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Occad the model"""
        self.q_network.load(filename)
        self._update_target()