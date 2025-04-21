import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tinygrad.tensor import Tensor
from agent import DQNAgent
import time

def train(env_name='CartPole-v1', n_episodes=500, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """
    Train a DQN agent.
    
    Args:
        env_name (str): Gymnasium environment name
        n_episodes (int): Number of episodes
        max_t (int): Maximum number of timesteps per episode
        eps_start (float): Starting value of epsilon for epsilon-greedy action selection
        eps_end (float): Minimum value of epsilon
        eps_decay (float): Multiplicative factor for decreasing epsilon
        
    Returns:
        list: Scores from each episode
    """
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    
    scores = []
    scores_window = []  # Last 100 scores
    eps = eps_start
    
    print("Training Q-Learning agent on", env_name)
    print(f"State size: {state_size}, Action size: {action_size}")
    
    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        score = 0
        
        for t in range(max_t):
            # Select and perform an action
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Update the agent
            agent.step(state, action, reward, next_state, done)
            
            state = next_state
            score += reward
            
            if done:
                break
        
        # Save score and decrease epsilon
        scores.append(score)
        scores_window.append(score)
        eps = max(eps_end, eps_decay * eps)
        
        # Print progress
        if i_episode % 20 == 0:
            mean_score = np.mean(scores_window[-100:])
            print(f"Episode {i_episode}/{n_episodes}, Average Score: {mean_score:.2f}, Epsilon: {eps:.2f}")
        
        # Save model if we reach target score (>= 195.0 over 100 episodes for CartPole-v1)
        if i_episode >= 100 and np.mean(scores_window) >= 195.0:
            print(f"\nEnvironment solved in {i_episode} episodes! Average Score: {np.mean(scores_window):.2f}")
            agent.save(f'checkpoint_{env_name}.npy')
            break
    
    # Save final model regardless
    agent.save(f'final_model_{env_name}.npy')
    
    # Plot scores
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.title(f'Q-Learning Training on {env_name}')
    plt.savefig('training_scores.png')
    
    return scores

if __name__ == "__main__":
    scores = train()
    print("Training complete!")