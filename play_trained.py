"""
Play the trained Lunar Lander agent locally with visualization
"""
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

device = torch.device("cpu")

# Neural Network
class ActorCritic(nn.Module):
    def __init__(self, obs_dim=8, act_dim=2, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.mu_head = nn.Linear(hidden, act_dim)
        self.v_head = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.ones(act_dim) * -0.5)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mu = torch.tanh(self.mu_head(x))
        value = self.v_head(x).squeeze(-1)
        std = torch.exp(self.log_std.clamp(-20, 2))
        return mu, std, value

# Observation normalization
class RunningMeanStd:
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
    
    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)

# Load model
print("Loading trained model...")
model = ActorCritic().to(device)
obs_rms = RunningMeanStd(8)

checkpoint = torch.load("lunar_lander_actor_critic_v3.pth", map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
obs_rms.mean = checkpoint['obs_rms_mean']
obs_rms.var = checkpoint['obs_rms_var']
model.eval()

print("Model loaded! Starting episodes...\n")

# Create environment with rendering
env = gym.make("LunarLanderContinuous-v3", render_mode="human")

# Play 5 episodes
for episode in range(5):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    print(f"Episode {episode + 1}/5")
    
    while not done:
        # Normalize observation and get action
        obs_norm = obs_rms.normalize(obs)
        obs_t = torch.from_numpy(obs_norm).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            mu, std, value = model(obs_t)
            action = mu.cpu().numpy()[0]
        
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)
        
        # Step environment
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
    
    print(f"  Reward: {total_reward:.1f} | Steps: {steps}\n")

env.close()
print("Done! Close the window to exit.")
