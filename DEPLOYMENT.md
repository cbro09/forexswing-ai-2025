# ForexSwing AI 2025 - VM Deployment Guide

## Overview

This guide will help you deploy ForexSwing AI 2025 to a Virtual Machine (VM) for continuous operation with:
- ✅ Automated market data collection
- ✅ News sentiment analysis with Gemini AI
- ✅ Enhanced LSTM predictions
- ✅ 24/7 operation with automatic restarts
- ✅ Docker containerization for easy deployment

---

## Prerequisites

### VM Requirements
- **OS**: Ubuntu 20.04/22.04 or Debian 11/12
- **RAM**: Minimum 4GB (8GB recommended)
- **CPU**: 2+ cores
- **Storage**: 20GB minimum (SSD recommended)
- **Network**: Stable internet connection

### Required API Keys
1. **Alpha Vantage** (Free): https://www.alphavantage.co/support/#api-key
   - Used for market data and news
2. **Google Gemini** (Free): https://makersuite.google.com/app/apikey
   - Used for AI market analysis
3. **News API** (Optional): https://newsapi.org/register
   - Additional news source

---

## Quick Start Deployment

### Option 1: Automated Setup (Recommended)

```bash
# 1. SSH into your VM
ssh user@your-vm-ip

# 2. Copy the project to your VM
# (Use scp, git clone, or your preferred method)
scp -r forexswing-ai-2025 user@your-vm-ip:/home/user/

# 3. Run the automated setup script
cd forexswing-ai-2025
chmod +x deploy/setup_vm.sh
./deploy/setup_vm.sh

# 4. Configure your API keys
nano .env

# 5. Start the services
sudo systemctl start forexbot
```

### Option 2: Manual Setup

```bash
# 1. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# 2. Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 3. Clone/copy the project
cd /opt
sudo mkdir forexswing-ai-2025
sudo chown $USER:$USER forexswing-ai-2025
cd forexswing-ai-2025

# 4. Create .env file
cp .env.example .env
nano .env  # Add your API keys

# 5. Build and start
docker-compose build
docker-compose up -d
```

---

## Configuration

### Environment Variables (.env)

Edit your `.env` file with your API keys:

```bash
# Required
ALPHA_VANTAGE_KEY=your_key_here
GOOGLE_API_KEY=your_gemini_key_here

# Optional
NEWS_API_KEY=your_news_api_key
TELEGRAM_BOT_TOKEN=your_telegram_token
TELEGRAM_CHAT_ID=your_chat_id

# Trading Settings
MAX_OPEN_TRADES=3
STAKE_AMOUNT=50
DRY_RUN=true
INITIAL_WALLET=1000
```

### Trading Pairs

Currently configured pairs (edit in `src/data/market_data_collector.py`):
- EUR/USD
- GBP/USD
- USD/JPY
- AUD/USD
- USD/CAD
- NZD/USD
- USD/CHF

---

## System Architecture

### Services

1. **forexbot**: Main trading bot with AI analysis API
   - Port: 8082
   - Provides real-time analysis via REST API

2. **data-collector**: Automated data collection
   - Runs every 4 hours for news updates
   - Daily full market data collection at 00:00 UTC

3. **model-trainer**: Model training (optional)
   - Run manually when you have enough data
   - Improves prediction accuracy

### Data Flow

```
Market Data (Alpha Vantage/Yahoo)
    ↓
Technical Indicators (JAX-optimized)
    ↓
News Sentiment (Gemini AI) ──→ Enhanced Features ──→ LSTM Model ──→ Trading Signal
    ↓
Historical Storage
```

---

## Initial Data Collection

Before the bot can make accurate predictions, collect historical data:

```bash
# One-time full data collection
docker-compose run data-collector python src/data/market_data_collector.py --once

# This will:
# 1. Download 2 years of historical data for all pairs
# 2. Calculate technical indicators
# 3. Fetch latest news sentiment
# 4. Save everything to data/MarketData and data/News
```

Expected runtime: 15-20 minutes (due to API rate limits)

---

## Model Training

### Initial Training

After collecting data, train your model:

```bash
# Train the model
docker-compose --profile training run model-trainer

# This will:
# 1. Load all market data and news
# 2. Create enhanced feature set
# 3. Train LSTM model for ~50 epochs
# 4. Save best model to data/models/enhanced_forex_ai.pth
```

Expected training time: 30-60 minutes (depending on CPU)

### Retraining

Retrain weekly or monthly with new data:

```bash
# Stop services
docker-compose down

# Collect latest data
docker-compose run data-collector python src/data/market_data_collector.py --once

# Retrain model
docker-compose --profile training run model-trainer

# Restart services
docker-compose up -d
```

---

## Monitoring & Management

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f forexbot
docker-compose logs -f data-collector

# System logs
sudo journalctl -u forexbot -f
```

### Check Status

```bash
# Docker containers
docker-compose ps

# Systemd service
sudo systemctl status forexbot

# API health check
curl http://localhost:8082/api/status
```

### Service Management

```bash
# Start
sudo systemctl start forexbot

# Stop
sudo systemctl stop forexbot

# Restart
sudo systemctl restart forexbot

# Enable on boot
sudo systemctl enable forexbot

# Disable on boot
sudo systemctl disable forexbot
```

---

## API Usage

### Get Analysis for Currency Pair

```bash
curl "http://your-vm-ip:8082/api/analyze?pair=EUR/USD"
```

Response:
```json
{
  "pair": "EUR/USD",
  "action": "BUY",
  "confidence": 0.67,
  "score": 0.45,
  "risk_level": "MEDIUM",
  "components": {
    "lstm": "BUY 62% [OK]",
    "gemini": "bullish 60% [OK]",
    "news": "+0.20 (5 articles) [OK]"
  },
  "data_quality": "3/3 models active",
  "timestamp": "2025-01-19T12:00:00",
  "processing_time": "0.234s"
}
```

### Get Supported Pairs

```bash
curl "http://your-vm-ip:8082/api/pairs"
```

### System Status

```bash
curl "http://your-vm-ip:8082/api/status"
```

---

## Improving Prediction Accuracy

### 1. Collect More Data

The model improves with more historical data:

```bash
# Let data collector run for 1-2 weeks
# Then retrain model
```

### 2. Add More News Sources

Edit `src/integrations/news_sentiment_analyzer.py` to add additional news APIs.

### 3. Feature Engineering

Modify `src/core/models/optimized_forex_lstm.py` to add:
- Economic indicators
- Correlation features
- Market regime detection

### 4. Hyperparameter Tuning

Edit `src/training/enhanced_model_trainer.py`:
```python
self.hidden_size = 256  # Increase model capacity
self.num_epochs = 100   # Train longer
self.learning_rate = 0.0005  # Adjust learning rate
```

### 5. Ensemble Models

Train multiple models and average predictions for stability.

---

## Troubleshooting

### Issue: Data Collection Fails

```bash
# Check API keys
cat .env | grep ALPHA_VANTAGE_KEY

# Check API quota
curl "https://www.alphavantage.co/query?function=FX_DAILY&from_symbol=EUR&to_symbol=USD&apikey=YOUR_KEY"

# Use manual data collection
docker-compose run data-collector python src/data/market_data_collector.py --once
```

### Issue: Model Not Loading

```bash
# Check if model file exists
ls -lh data/models/enhanced_forex_ai.pth

# Retrain model
docker-compose --profile training run model-trainer
```

### Issue: Gemini API Not Working

```bash
# Verify Gemini key
# The bot will still work with LSTM and news sentiment
# Update key in .env and restart

docker-compose restart
```

### Issue: Low Accuracy

1. Collect more training data (2+ years recommended)
2. Ensure news sentiment is working
3. Check data quality: `ls -lh data/MarketData/`
4. Retrain with more epochs
5. Wait for model to learn from live data

---

## Security Best Practices

1. **Never commit .env to git**
   ```bash
   # Already in .gitignore
   echo ".env" >> .gitignore
   ```

2. **Use firewall rules**
   ```bash
   sudo ufw allow 8082/tcp  # API port
   sudo ufw enable
   ```

3. **Regular updates**
   ```bash
   sudo apt-get update && sudo apt-get upgrade -y
   docker-compose pull
   docker-compose up -d
   ```

4. **Backup data regularly**
   ```bash
   tar -czf forexbot-backup-$(date +%Y%m%d).tar.gz data/
   ```

---

## Performance Optimization

### For CPU-Limited VMs

```python
# In src/training/enhanced_model_trainer.py
self.batch_size = 16  # Reduce from 32
self.num_workers = 0  # Disable multiprocessing
```

### For Memory-Limited VMs

```yaml
# In docker-compose.yml, add memory limits
services:
  forexbot:
    mem_limit: 2g
```

### For Faster Inference

```python
# In ForexBot.py
self.model.set_cache_enabled(True)  # Enable caching
```

---

## Scaling to Multiple VMs

### Load Balancer Setup

```
                    ┌─────────────┐
                    │Load Balancer│
                    └──────┬──────┘
              ┌────────────┼────────────┐
         ┌────▼────┐  ┌────▼────┐  ┌────▼────┐
         │  VM 1   │  │  VM 2   │  │  VM 3   │
         │ForexBot │  │ForexBot │  │ForexBot │
         └─────────┘  └─────────┘  └─────────┘
              │            │            │
              └────────────┴────────────┘
                         │
                   ┌─────▼──────┐
                   │Shared Data │
                   │   Storage  │
                   └────────────┘
```

---

## Cost Estimation

### Cloud VM Pricing (Monthly)

- **AWS EC2 t3.medium**: $30-35/month
- **Google Cloud e2-medium**: $25-30/month
- **DigitalOcean 4GB**: $24/month
- **Linode 4GB**: $24/month

### API Costs (Free Tiers)

- **Alpha Vantage**: Free (500 calls/day)
- **Google Gemini**: Free (60 requests/minute)
- **News API**: Free (100 requests/day)

**Total**: ~$25-35/month for basic setup

---

## Next Steps

1. ✅ Deploy to VM
2. ✅ Collect historical data
3. ✅ Train initial model
4. ✅ Monitor for 1 week
5. ✅ Retrain with collected data
6. ✅ Integrate with broker API (MT5, OANDA, etc.)
7. ✅ Set up alerts and monitoring
8. ✅ Backtest strategies

---

## Support & Resources

- **Documentation**: Check CLAUDE.md for project overview
- **Logs**: `data/logs/` directory
- **Issues**: Monitor Docker logs for errors
- **Updates**: Pull latest code and rebuild

---

## Success Checklist

- [ ] VM provisioned with required specs
- [ ] Docker and Docker Compose installed
- [ ] API keys configured in .env
- [ ] Initial data collected successfully
- [ ] Model trained and saved
- [ ] Services running and auto-restarting
- [ ] API accessible from external IP
- [ ] News sentiment working
- [ ] Gemini integration active
- [ ] Automated data collection scheduled
- [ ] Monitoring and logs accessible

---

**Your ForexSwing AI 2025 bot is now running 24/7 with continuous learning!**
