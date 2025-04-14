from replay_buffer import *
from exploration_strategy import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Callable
import gymnasium as gym
from collections import deque

class FCQ(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dims = (32,32),
                 activation_fc = F.relu):
        super(FCQ, self).__init__()
        # create input layer, activation, and layers
        self.input_layer = nn.Linear(input_dim, hidden_dims[0]) # take in state and connect x number of states to 32 neurons
        self.activation_fc = activation_fc
        self.hidden_layers = nn.ModuleList()
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        for i in range(len(hidden_dims)-1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        
    
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
        self.to(self.device)
        print(f"Using {self.device} as device")
    # format 
    def format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x,
                                device = self.device,
                                dtype=torch.float32)
            if x.dim() == 1:
                x = x.unsqueeze(0) # add a batch dimension 
        return x
    
    def forward(self, state):
        x = self.format(state)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        x = self.output_layer(x)
        return x
    
### MAIN TRAINING LOOP BELOW

# SETUP GYM ENVIRONMENT + DIMENSION VALUES
env = gym.make("CartPole-v0", render_mode="rgb_array")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n
print(f"Environment CartPole is set up. Obs dim = {obs_dim} and act_dim = {act_dim}")

# LOGGING ARRAYS
rewards_per_episode = []
losses_per_episode = []
epsilon_per_episode = []

# VARIABLES
n_episodes = 1000
max_buffer = 10000
min_replays = 500
gamma = 0.99
learning_rate = 1e-3
decay_rate = 1000
update_freq = 5
global_step = 0

# SET UP CLASSES
q_network = FCQ(obs_dim, act_dim, hidden_dims=(64,64,64))
replay_buffer = ReplayBuffer(obs_dim, act_dim, size=max_buffer)
action_strategy = EpsilonStrategy(1.0, 0.05, decay_rate=decay_rate)
optimizer = torch.optim.Adam(q_network.parameters(), lr=learning_rate)
target_network = FCQ(obs_dim, act_dim, hidden_dims=(64,64,64))
target_network.load_state_dict(q_network.state_dict())

# INITIATE LOOP THROUGH EPISODES
for episode in range(n_episodes):
    obs, info = env.reset() # always reset the environment at start of new episode 
    done = False # done flag always initially False
    episode_reward = 0
    episode_losses = []
    # inner while condition for each episode 
    
    while not done:
        q_values = q_network(obs) # run observation through network to obtain q_values
        action = action_strategy.select_action(q_values)
        next_obs, rew, terminated, truncated, _ = env.step(action) 
        done = terminated or truncated
        replay_buffer.store(obs, next_obs, action, rew, done)
        obs = next_obs
        episode_reward += rew
        
        # training loop if we have enough samples
        if len(replay_buffer) >= min_replays:
            batch = replay_buffer.sample_batch(batch_size=128)
            state, next_state, acts, rews, dones = batch['obs'], batch['obs2'], batch['act'], batch['rew'], batch['done']
            q_vals = q_network(state)
            current_val = q_vals.gather(1, acts.unsqueeze(-1)).squeeze()
            
            # compute target without tracking gradient
            with torch.no_grad():
                next_best_action = q_network(next_state).argmax(1)
                q_val_next = target_network(next_state)
                q_max = q_val_next.gather(1, next_best_action.unsqueeze(-1)).squeeze()
                target_val = rews + gamma * q_max * (1 - dones)
                
            # compute loss
            loss = F.mse_loss(current_val, target_val)
            # step through
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            
            if global_step % update_freq == 0:
                target_network.load_state_dict(q_network.state_dict())
            episode_losses.append(loss.item())
            
    # Track metrics at episode end
    rewards_per_episode.append(episode_reward)
    avg_loss = np.mean(episode_losses) if episode_losses else 0
    losses_per_episode.append(avg_loss)
    epsilon_per_episode.append(action_strategy.epsilon)
    if (episode+1) % 10 == 0:
        avg_last_100 = np.mean(rewards_per_episode[-100:]) if len(rewards_per_episode) >= 100 else np.mean(rewards_per_episode)
        print(f"Episode {episode+1}: Reward={episode_reward:.2f}, AvgLoss={avg_loss:.4f}, Epsilon={action_strategy.epsilon:.4f}, "
          f"Avg100={avg_last_100:.2f}")
    # Save the model if it did good - can leave it if too many models are saved or choose to save if only appropriate reward is received at a 10th or even 50th episode
    if episode_reward > 195:
        torch.save(q_network.state_dict(), f"cartpole_qnet_ep{episode}.pt")
        print(f"Saved model at episode {episode} with reward {episode_reward:.2f}")
        
    if len(rewards_per_episode) >= 100:
        avg_last_100 = np.mean(rewards_per_episode[-100:])
        if avg_last_100 >= 195:
            print(f"Solved in {episode} episodes! Avg of last 100 = {avg_last_100:.2f}")
            break
      
def plot_training(rewards, losses, epsilons):
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))

    axs[0].plot(rewards)
    axs[0].set_title("Episode Reward")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Reward")

    axs[1].plot(losses)
    axs[1].set_title("Episode Loss")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Loss")

    axs[2].plot(epsilons)
    axs[2].set_title("Epsilon")
    axs[2].set_xlabel("Episode")
    axs[2].set_ylabel("Epsilon Value")

    plt.tight_layout()
    plt.show()

plot_training(rewards_per_episode, losses_per_episode, epsilon_per_episode)
    