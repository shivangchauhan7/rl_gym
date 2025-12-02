# Deploying to Render.com

## Quick Deployment Steps

### Option 1: Using Render Dashboard (Recommended)

1. **Push your code to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Go to Render Dashboard**
   - Visit https://render.com and sign in
   - Click "New +" button → "Web Service"

3. **Connect Your Repository**
   - Connect your GitHub account
   - Select your repository
   - Click "Connect"

4. **Configure the Service**
   - **Name**: rl-cartpole (or your preferred name)
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --timeout 300 --workers 1`
   - **Instance Type**: Free (or your preference)

5. **Deploy**
   - Click "Create Web Service"
   - Wait for the build and deployment to complete (may take 5-10 minutes)
   - Your app will be available at: `https://your-service-name.onrender.com`

### Option 2: Using render.yaml (Blueprint)

The `render.yaml` file is already included in your project. When creating a new service:
- Choose "Blueprint" instead of "Web Service"
- Select your repository
- Render will automatically detect and use the `render.yaml` configuration

## Important Notes

- **Free Tier**: Services on the free tier spin down after 15 minutes of inactivity
- **Cold Starts**: First request after inactivity may take 30-60 seconds
- **Memory**: The free tier has 512MB RAM - sufficient for this app
- **Checkpoints**: Make sure all your model checkpoint files are committed to the repo

## Files Required for Deployment

✅ `app.py` - Your Flask application
✅ `requirements.txt` - Python dependencies
✅ `Procfile` - Process configuration
✅ `runtime.txt` - Python version specification
✅ `render.yaml` - Render service configuration
✅ `templates/index.html` - Frontend template
✅ `checkpoints/` - Model checkpoint files
✅ `lunar_lander_actor_critic_v3.pth` - Trained model

## Troubleshooting

### Build Fails
- Check that all dependencies in `requirements.txt` are compatible
- Verify Python version matches `runtime.txt`

### App Crashes
- Check the logs in Render dashboard
- Ensure all model files are present in the repository
- Verify the start command is correct

### Timeout Issues
- The app uses `--timeout 300` to handle long-running simulations
- If needed, this can be increased in the Procfile or render.yaml

## Testing Locally Before Deployment

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py

# Or use gunicorn (production server)
gunicorn app:app --timeout 300 --workers 1
```

Visit `http://localhost:5000` to test.
