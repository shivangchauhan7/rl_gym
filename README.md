# LunarLander AI Web App

A web interface to watch a trained PPO agent land the lunar module.

## Local Testing

```bash
pip install -r requirements.txt
python app.py
```

Visit http://localhost:5000

## Deploy to Heroku

1. Install Heroku CLI: https://devcenter.heroku.com/articles/heroku-cli

2. Login to Heroku:
```bash
heroku login
```

3. Create a new Heroku app:
```bash
heroku create your-app-name
```

4. Add buildpack for Python:
```bash
heroku buildpacks:set heroku/python
```

5. Deploy:
```bash
git init
git add .
git commit -m "Initial commit"
git push heroku main
```

6. Open your app:
```bash
heroku open
```

## Files

- `app.py` - Flask web server
- `templates/index.html` - Web interface
- `lunar_lander_actor_critic_v3.pth` - Trained model
- `requirements.txt` - Python dependencies
- `Procfile` - Heroku configuration
- `runtime.txt` - Python version

## How it Works

When you click "Start 100 Episodes", the server:
1. Creates a LunarLander environment
2. Runs the trained agent for 100 episodes
3. Streams video frames to the browser
4. Updates statistics in real-time

The agent uses observation normalization and a 128-unit neural network trained with PPO.
