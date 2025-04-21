import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from agent import DQNAgent
import time

def plot_scores(scores):
    """Plot the training scores"""
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.title('Q-Learning Training Scores')
    plt.savefig('training_scores.png')
    plt.show()

def visualize_agent(model_path, env_name='CartPole-v1', n_episodes=3):
    """
    Visualize a trained agent.
    
    Args:
        model_path (str): Path to the saved model
        env_name (str): Gymnasium environment name
        n_episodes (int): Number of episodes to visualize
    """
    env = gym.make(env_name, render_mode="human")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    agent.load(model_path)
    
    for i in range(n_episodes):
        state, _ = env.reset()
        score = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            score += reward
            time.sleep(0.01)  # Slow down visualization
            
        print(f"Episode {i+1}: Score = {score}")
    
    env.close()

if __name__ == "__main__":
    # You can load saved scores or run visualization
    # Example to visualize a trained agent:
    visualize_agent('final_model_CartPole-v1.npy')