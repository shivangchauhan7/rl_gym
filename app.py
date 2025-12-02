"""
Flask web app with Learning Progress Visualization
Shows agent performance at different training stages (0%, 20%, 40%, 60%, 80%, 100%)
"""
from flask import Flask, render_template, jsonify, Response, request
import gymnasium as gym
from gymnasium.envs.registration import register
import torch
import torch.nn as nn
import numpy as np
import cv2
from threading import Thread, Lock
import base64
import time
import os
import glob

app = Flask(__name__)
device = torch.device("cpu")

# Check if Box2D is available
try:
    test_env = gym.make("LunarLanderContinuous-v3")
    test_env.close()
    HAS_BOX2D = True
    print("✓ Box2D available - using real LunarLander environment")
except Exception as e:
    HAS_BOX2D = False
    print(f"⚠ Box2D not available: {e}")
    print("  Using dummy environment for Heroku deployment")
    
    # Register dummy environment
    class DummyLunarLander:
        def __init__(self, render_mode=None):
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
            self.render_mode = render_mode
            self.steps = 0
            self.state = np.zeros(8, dtype=np.float32)
            
        def reset(self, seed=None):
            self.steps = 0
            self.state = np.random.randn(8).astype(np.float32) * 0.1
            return self.state.copy(), {}
        
        def step(self, action):
            self.steps += 1
            # Simulate some dynamics
            self.state += np.random.randn(8).astype(np.float32) * 0.05
            self.state = np.clip(self.state, -1, 1)
            
            # Reward based on dummy criteria
            reward = -abs(self.state[1]) * 10  # Penalty for being far from center
            terminated = self.steps >= 500
            truncated = False
            return self.state.copy(), reward, terminated, truncated, {}
        
        def render(self):
            # Return a simple placeholder image
            img = np.ones((400, 600, 3), dtype=np.uint8) * 30
            # Draw simple representation
            cv2.putText(img, "Lunar Lander Simulation", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, "(Box2D not available on Heroku)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            cv2.putText(img, f"Step: {self.steps}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
            # Draw lander representation
            cx, cy = 300, 200
            cv2.circle(img, (cx, cy), 20, (100, 200, 100), -1)
            cv2.rectangle(img, (cx-30, cy+20), (cx+30, cy+40), (150, 150, 150), -1)
            return img
        
        def close(self):
            pass
    
    try:
        register(
            id='LunarLanderContinuous-v3',
            entry_point=lambda render_mode=None: DummyLunarLander(render_mode=render_mode),
        )
    except:
        pass  # Already registered

# Neural Network - Same architecture for all checkpoints
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

class RunningMeanStd:
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
    
    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)

# Load available checkpoints
def load_available_checkpoints():
    checkpoints = {}
    
    # Main trained model
    if os.path.exists('lunar_lander_actor_critic_v3.pth'):
        checkpoints['100'] = {
            'path': 'lunar_lander_actor_critic_v3.pth',
            'label': '100% Trained (Full)',
            'description': 'Fully trained agent - expert performance'
        }
    
    # Training checkpoints
    if os.path.exists('checkpoints'):
        for f in sorted(glob.glob('checkpoints/model_*.pth')):
            filename = os.path.basename(f)
            if 'pct' in filename:
                pct = filename.split('_')[1].replace('pct.pth', '')
                checkpoints[pct] = {
                    'path': f,
                    'label': f'{pct}% Trained',
                    'description': f'Agent after {pct}% of training'
                }
    
    return checkpoints

available_checkpoints = load_available_checkpoints()

# Global state
current_model = None
current_obs_rms = None
current_checkpoint_id = '100'
current_frame = None
frame_lock = Lock()
stats = {
    'episode': 0, 
    'reward': 0, 
    'total_episodes': 0, 
    'avg_reward': 0, 
    'running': False, 
    'training_stage': 'Ready',
    'phase': 'training'  # 'training' or 'trained'
}
stats_lock = Lock()

def load_model(checkpoint_path):
    """Load model from checkpoint"""
    obs_rms = RunningMeanStd(8)
    
    # Use same architecture for all checkpoints
    model = ActorCritic().to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    obs_rms.mean = checkpoint['obs_rms_mean']
    obs_rms.var = checkpoint['obs_rms_var']
    model.eval()
    
    return model, obs_rms

# Load default model (100%)
current_model, current_obs_rms = load_model('lunar_lander_actor_critic_v3.pth')

def run_training_progression():
    """Show training progression using actual checkpoint models"""
    global current_frame, stats
    
    env = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")
    
    with stats_lock:
        stats['running'] = True
        stats['phase'] = 'training'
    
    # Training stages with descriptions
    training_stages = [
        ('0', 'Just started training - Random actions'),
        ('20', 'Learning basic control'),
        ('40', 'Understanding landing mechanics'),
        ('60', 'Improving stability'),
        ('80', 'Nearly expert level'),
        ('100', 'Fully trained - Expert performance!')
    ]
    
    print("\n🎓 Starting Training Visualization...")
    print("Showing progress from 0% → 100% training with real checkpoints\n")
    
    # Show progression through training stages (2 episodes each)
    for stage_pct, stage_desc in training_stages:
        print(f"  📊 {stage_pct}% Training: {stage_desc}")
        
        with stats_lock:
            stats['training_stage'] = f'{stage_pct}% Training'
        
        # Load the checkpoint for this stage
        checkpoint_path = available_checkpoints[stage_pct]['path']
        stage_model, stage_obs_rms = load_model(checkpoint_path)
        
        # Run 2 episodes at this stage
        stage_rewards = []
        for ep in range(2):
            obs, _ = env.reset()
            done = False
            ep_reward = 0
            
            while not done:
                # Render frame
                frame = env.render()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                _, buffer = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                
                with frame_lock:
                    current_frame = frame_b64
                
                # Get action from current stage model
                obs_norm = stage_obs_rms.normalize(obs)
                obs_t = torch.from_numpy(obs_norm).float().unsqueeze(0).to(device)
                
                with torch.no_grad():
                    mu, std, _ = stage_model(obs_t)
                    action = mu.cpu().numpy()[0]
                
                # Step environment
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ep_reward += reward
                
                with stats_lock:
                    stats['reward'] = ep_reward
                
                # Real-time speed
                time.sleep(0.016)
        
            stage_rewards.append(ep_reward)
        
        avg_stage_reward = np.mean(stage_rewards)
        print(f"     → Episodes: {stage_rewards[0]:.1f}, {stage_rewards[1]:.1f} | Avg: {avg_stage_reward:.1f}")
    
    print(f"\n✅ Training Complete! Agent is now fully trained!\n")
    
    with stats_lock:
        stats['training_stage'] = '100% Trained - Expert!'
        stats['phase'] = 'trained'
        stats['episode'] = 0
        stats['total_episodes'] = 0
        stats['avg_reward'] = 0
    
    env.close()
    
    with stats_lock:
        stats['running'] = False

def run_trained_episodes():
    """Run 5 episodes with fully trained model"""
    global current_frame, stats
    
    print("\n🔄 Replaying Trained Agent (5 episodes)...")
    
    env = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")
    checkpoint_path = available_checkpoints['100']['path']
    trained_model, trained_obs_rms = load_model(checkpoint_path)
    
    with stats_lock:
        stats['running'] = True
        stats['phase'] = 'trained'
        stats['training_stage'] = '100% Trained - Expert!'
        stats['episode'] = 0
        stats['total_episodes'] = 0
        stats['avg_reward'] = 0
    
    all_rewards = []
    
    # Run 5 episodes
    for ep in range(5):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        
        with stats_lock:
            stats['episode'] = ep + 1
        
        print(f"  Starting episode {ep + 1}/5...")
        
        while not done:
            # Render frame
            frame = env.render()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            with frame_lock:
                current_frame = frame_b64
            
            # Get action
            obs_norm = trained_obs_rms.normalize(obs)
            obs_t = torch.from_numpy(obs_norm).float().unsqueeze(0).to(device)
            
            with torch.no_grad():
                mu, std, _ = trained_model(obs_t)
                action = mu.cpu().numpy()[0]
            
            # Step environment
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            
            with stats_lock:
                stats['reward'] = ep_reward
            
            # Real-time speed
            time.sleep(0.016)
        
        all_rewards.append(ep_reward)
        
        with stats_lock:
            stats['total_episodes'] = ep + 1
            stats['avg_reward'] = np.mean(all_rewards)
        
        print(f"  Episode {ep + 1}/5: Reward = {ep_reward:.1f}")
    
    print(f"\n📈 Replay Complete! Average Reward: {np.mean(all_rewards):.1f}\n")
    
    env.close()
    
    with stats_lock:
        stats['running'] = False

def run_episodes():
    """Dispatcher - run training progression or trained episodes"""
    with stats_lock:
        phase = stats.get('phase', 'training')
    
    if phase == 'training':
        run_training_progression()
    else:
        run_trained_episodes()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/checkpoints')
def get_checkpoints():
    """Return available training checkpoints"""
    return jsonify(available_checkpoints)

@app.route('/start')
def start():
    with stats_lock:
        if not stats['running']:
            stats['running'] = True
            Thread(target=run_episodes, daemon=True).start()
            return jsonify({'status': 'started'})
        return jsonify({'status': 'already_running'})

@app.route('/replay')
def replay():
    """Replay trained model episodes"""
    with stats_lock:
        if not stats['running']:
            stats['phase'] = 'trained'
            stats['running'] = True
            Thread(target=run_trained_episodes, daemon=True).start()
            return jsonify({'status': 'replaying'})
        return jsonify({'status': 'already_running'})

@app.route('/frame')
def frame():
    with frame_lock:
        if current_frame:
            return jsonify({'frame': current_frame})
        return jsonify({'frame': None})

@app.route('/stats')
def get_stats():
    with stats_lock:
        return jsonify(stats)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    print("\n" + "="*60)
    print("🚀 Lunar Lander RL Learning Visualization")
    print("="*60)
    print(f"\nAvailable checkpoints: {len(available_checkpoints)}")
    for checkpoint_id, info in sorted(available_checkpoints.items(), key=lambda x: int(x[0])):
        print(f"  • {info['label']}")
    
    if len(available_checkpoints) == 1:
        print("\n⚠️  Only 100% checkpoint available.")
        print("Run: python3 train_with_checkpoints.py")
        print("This will create checkpoints showing learning progress.")
    
    print(f"\n🌐 Server starting at: http://0.0.0.0:{port}")
    print("="*60 + "\n")
    
    # Disable Flask request logging
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    # Use debug=False for production
    app.run(debug=False, host='0.0.0.0', port=port)
