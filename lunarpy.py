import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============= ENV =============
env = gym.make("LunarLanderContinuous-v3", render_mode=None)
obs_dim = env.observation_space.shape[0]   # 8
act_dim = env.action_space.shape[0]        # 2

print("obs_dim:", obs_dim, "act_dim:", act_dim)

# ============= MODEL =============
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        # Balanced network size for stable training
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.mu_head = nn.Linear(hidden, act_dim)
        self.v_head  = nn.Linear(hidden, 1)

        # Orthogonal initialization for all layers
        for layer in [self.fc1, self.fc2]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0)
        
        nn.init.orthogonal_(self.mu_head.weight, gain=0.01)
        nn.init.constant_(self.mu_head.bias, 0)
        nn.init.orthogonal_(self.v_head.weight, gain=1.0)
        nn.init.constant_(self.v_head.bias, 0)

        # Balanced std for exploration
        self.log_std = nn.Parameter(torch.ones(act_dim) * -0.5)  # std ~ 0.6

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mu = torch.tanh(self.mu_head(x))  # Bound output to [-1, 1]
        value = self.v_head(x).squeeze(-1)
        std = torch.exp(self.log_std.clamp(-20, 2))  # Clamp std for stability
        return mu, std, value

model = ActorCritic(obs_dim, act_dim).to(device)

# ============= GAE =============
def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    T = rewards.size(0)
    advantages = torch.zeros(T, device=device)

    values_ext = torch.cat([values, last_value.unsqueeze(0)], dim=0)

    gae = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * values_ext[t+1] * mask - values_ext[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae

    returns = advantages + values
    return returns, advantages

# ============= PPO SETTINGS =============
total_timesteps = 1_000_000       # balanced training duration
rollout_steps   = 2048
ppo_epochs      = 10              # more epochs for thorough optimization
minibatch_size  = 64              # smaller batches for better gradients
max_episode_steps = 1000          # prevent extremely long episodes
gamma           = 0.99            # standard gamma
lam             = 0.95            # standard lambda
clip_eps        = 0.2
learning_rate   = 3e-4
value_coef      = 0.5
entropy_coef    = 0.01            # exploration
max_grad_norm   = 0.5

# Learning rate annealing
def get_lr(update, num_updates):
    frac = 1.0 - (update - 1.0) / num_updates
    return learning_rate * frac

# Entropy coefficient decay for stability
def get_entropy_coef(update, num_updates):
    # Gradual decay: 0.01 → 0.001 over training
    frac = 1.0 - (update - 1.0) / num_updates
    return 0.001 + 0.009 * frac

# Observation normalization
class RunningMeanStd:
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = 1e-4
    
    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)
    
    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
    
    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)

obs_rms = RunningMeanStd(obs_dim)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_updates = total_timesteps // rollout_steps

reward_history = []
episode_rewards = []

obs, _ = env.reset()

# ============= PPO TRAINING LOOP =============
for update in range(1, num_updates + 1):

    # ---- Rollout storage ----
    obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf = [], [], [], [], [], []
    steps_collected = 0

    while steps_collected < rollout_steps:
        obs_norm = obs_rms.normalize(obs)
        obs_buf.append(obs_norm)

        obs_t = torch.from_numpy(obs_norm).float().unsqueeze(0).to(device)
        with torch.no_grad():
            mu, std, value = model(obs_t)
            dist = Normal(mu, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)

        action_np = action.squeeze(0).cpu().numpy()
        action_np = np.clip(action_np, -1.0, 1.0)

        next_obs, reward, terminated, truncated, _ = env.step(action_np)
        done = terminated or truncated

        act_buf.append(action_np)
        logp_buf.append(log_prob.item())
        rew_buf.append(reward)
        done_buf.append(float(done))
        val_buf.append(value.squeeze(0).item())

        steps_collected += 1

        # episodic rewards tracker
        if len(episode_rewards) == 0:
            episode_rewards.append(reward)
        else:
            episode_rewards[-1] += reward

        obs = next_obs
        if done:
            obs, _ = env.reset()
            episode_rewards.append(0.0)

    # ---- Convert buffers ----
    obs_tensor = torch.from_numpy(np.array(obs_buf, dtype=np.float32)).to(device)
    act_tensor = torch.from_numpy(np.array(act_buf, dtype=np.float32)).to(device)
    logp_old   = torch.from_numpy(np.array(logp_buf, dtype=np.float32)).to(device)
    rewards    = torch.from_numpy(np.array(rew_buf, dtype=np.float32)).to(device)
    dones      = torch.from_numpy(np.array(done_buf, dtype=np.float32)).to(device)
    values     = torch.from_numpy(np.array(val_buf, dtype=np.float32)).to(device)

    # Update observation statistics
    obs_rms.update(np.array(obs_buf, dtype=np.float32))

    # ---- Bootstrap last value ----
    obs_norm = obs_rms.normalize(obs)
    obs_t = torch.from_numpy(obs_norm).float().unsqueeze(0).to(device)
    with torch.no_grad():
        _, _, last_value = model(obs_t)
        last_value = last_value.squeeze(0)

    # ---- No reward normalization (LunarLander rewards are already scaled) ----

    # ---- Compute GAE ----
    returns, advantages = compute_gae(rewards, values, dones, last_value, gamma, lam)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # ---- PPO Update ----
    # Update learning rate
    current_lr = get_lr(update, num_updates)
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    
    # Update entropy coefficient (decay for stability)
    current_entropy_coef = get_entropy_coef(update, num_updates)
    
    batch_size = rollout_steps
    indices = np.arange(batch_size)

    for epoch in range(ppo_epochs):
        np.random.shuffle(indices)
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_idx = indices[start:end]

            mb_obs   = obs_tensor[mb_idx]
            mb_acts  = act_tensor[mb_idx]
            mb_logp_old = logp_old[mb_idx]
            mb_ret   = returns[mb_idx]
            mb_adv   = advantages[mb_idx]

            mu, std, value = model(mb_obs)
            dist = Normal(mu, std)
            logp = dist.log_prob(mb_acts).sum(-1)
            entropy = dist.entropy().sum(-1)

            ratio = torch.exp(logp - mb_logp_old)

            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (mb_ret - value).pow(2).mean()
            entropy_loss = -entropy.mean()

            # Combined loss
            loss = (policy_loss + 
                    value_coef * value_loss + 
                    current_entropy_coef * entropy_loss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

    # ---- Logging ----
    if len(episode_rewards) > 1:
        finished = episode_rewards[:-1]
        reward_history.extend(finished)
        episode_rewards = episode_rewards[-1:]

    if update % 10 == 0 and len(reward_history) >= 10:
        avg10 = np.mean(reward_history[-10:])
        avg100 = np.mean(reward_history[-100:]) if len(reward_history) >= 100 else avg10
        print(f"Update {update}/{num_updates} | avg10: {avg10:.1f} | avg100: {avg100:.1f} | episodes: {len(reward_history)}")

# ============= SAVE MODEL =============
save_dict = {
    'model_state_dict': model.state_dict(),
    'obs_rms_mean': obs_rms.mean,
    'obs_rms_var': obs_rms.var,
    'obs_rms_count': obs_rms.count
}
torch.save(save_dict, "lunar_lander_actor_critic_v3.pth")
if len(reward_history) >= 100:
    final_avg = np.mean(reward_history[-100:])
    print(f"\nTraining complete! Final avg100 reward: {final_avg:.1f}")
    if final_avg >= 200:
        print("✓ Environment SOLVED! (avg100 >= 200)")
    else:
        print(f"Environment not yet solved. Need {200 - final_avg:.1f} more reward.")
print("Model saved!")

# ============= EVALUATION =============
print("\n=== Evaluating trained policy ===")
eval_env = gym.make("LunarLanderContinuous-v3", render_mode=None)

def get_action(mu_val):
    return np.clip(mu_val.squeeze(0).cpu().numpy(), -1.0, 1.0)

eval_rewards = []
for ep in range(10):
    obs, _ = eval_env.reset()
    done = False
    ep_reward = 0
    while not done:
        obs_norm = obs_rms.normalize(obs)
        obs_t = torch.from_numpy(obs_norm).float().unsqueeze(0).to(device)
        with torch.no_grad():
            mu, std, value = model(obs_t)
        action = get_action(mu)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
        ep_reward += reward
    eval_rewards.append(ep_reward)
    print(f"Eval episode {ep+1}: {ep_reward:.1f}")

eval_env.close()
print(f"\nAverage evaluation reward: {np.mean(eval_rewards):.1f} ± {np.std(eval_rewards):.1f}")
