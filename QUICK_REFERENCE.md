# ForexSwing AI 2025 - Quick Reference Card

## 🚀 Ultra-Quick Deployment (5 Minutes)

### On Your Windows Machine:
```bash
# 1. Prepare deployment package
setup_windows.bat
# Select: 4) Prepare for VM deployment
```

### On Your VM:
```bash
# 1. Copy files (from Windows)
scp -r deployment_package/* user@your-vm:/opt/forexswing-ai-2025/

# 2. SSH into VM
ssh user@your-vm

# 3. One-command setup
cd /opt/forexswing-ai-2025 && chmod +x deploy/setup_vm.sh quick_start.sh && ./deploy/setup_vm.sh

# 4. Add API keys
nano .env
# Add: ALPHA_VANTAGE_KEY and GOOGLE_API_KEY

# 5. Start everything
./quick_start.sh
# Select: 7) Collect data → 1) Start services
```

**Done! Bot running 24/7 on VM.**

---

## 📋 Essential Commands

### Quick Start Menu (Interactive)
```bash
./quick_start.sh
```

### Common Docker Operations
```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# View logs (live)
docker-compose logs -f

# Restart services
docker-compose restart

# Check status
docker-compose ps
```

### Data & Training
```bash
# Collect market data (one-time)
docker-compose run data-collector python src/data/market_data_collector.py --once

# Train/retrain model
docker-compose --profile training run model-trainer

# Monitor dashboard
python src/monitoring/dashboard.py http://your-vm-ip:8082
```

### API Usage
```bash
# Get analysis
curl "http://your-vm-ip:8082/api/analyze?pair=EUR/USD"

# List pairs
curl "http://your-vm-ip:8082/api/pairs"

# System status
curl "http://your-vm-ip:8082/api/status"
```

---

## 🔑 API Keys (All FREE)

| Service | Purpose | Get Key | Variable |
|---------|---------|---------|----------|
| **Alpha Vantage** | Market data + news | [alphavantage.co](https://www.alphavantage.co/support/#api-key) | `ALPHA_VANTAGE_KEY` |
| **Google Gemini** | AI analysis | [makersuite.google.com](https://makersuite.google.com/app/apikey) | `GOOGLE_API_KEY` |
| **News API** | Extra news (optional) | [newsapi.org](https://newsapi.org/register) | `NEWS_API_KEY` |

Add to `.env` file.

---

## 📊 System Architecture

```
Data Sources → Data Collector → Feature Engineering → Models → API
    ↓              (4h)              (JAX)           ↓      ↓
Market Data    Technical         Enhanced      LSTM + Gemini → Prediction
News Feed      Indicators        Features      + News       → BUY/SELL/HOLD
```

---

## 🎯 Expected Performance

| Metric | Value |
|--------|-------|
| **LSTM Accuracy** | 55.2% base |
| **With News** | 60-65% |
| **Processing Speed** | <0.5s |
| **API Response** | <1s |
| **Data Updates** | Every 4h |
| **Uptime** | 99.9% (auto-restart) |

---

## 💰 Monthly Costs

- **VM Hosting**: $24-35/month
- **API Calls**: FREE (within limits)
- **Total**: ~$25-35/month

---

## 🛠️ Troubleshooting One-Liners

```bash
# API not responding
docker-compose restart && docker-compose logs -f forexbot

# Data collection failed
docker-compose run data-collector python src/data/market_data_collector.py --once

# Model not loading
docker-compose --profile training run model-trainer

# Check disk space
df -h

# View system resources
docker stats

# Clean up old containers
docker system prune -a
```

---

## 📅 Maintenance Schedule

| Task | Frequency | Command |
|------|-----------|---------|
| Check logs | Daily | `docker-compose logs --tail=50` |
| Data backup | Weekly | `tar -czf backup-$(date +%Y%m%d).tar.gz data/` |
| Retrain model | Weekly/Monthly | `docker-compose --profile training run model-trainer` |
| Update system | Monthly | `git pull && docker-compose build && docker-compose up -d` |

---

## 🎯 Currency Pairs

Configured pairs (edit in `src/data/market_data_collector.py`):
- EUR/USD
- GBP/USD
- USD/JPY
- AUD/USD
- USD/CAD
- NZD/USD
- USD/CHF

---

## 📦 File Locations

| Data Type | Location |
|-----------|----------|
| Market data | `data/MarketData/*.csv` |
| News sentiment | `data/News/*.json` |
| Trained models | `data/models/*.pth` |
| Logs | `data/logs/*.log` |
| Configuration | `.env` |

---

## ⚡ Quick Tests

```bash
# Test API is running
curl http://localhost:8082/api/status

# Test analysis for EUR/USD
curl "http://localhost:8082/api/analyze?pair=EUR/USD" | python -m json.tool

# Test data collection
ls -lh data/MarketData/

# Test model exists
ls -lh data/models/

# Test news sentiment
ls -lh data/News/
```

---

## 🚨 Emergency Commands

```bash
# Stop everything
docker-compose down

# Start fresh
docker-compose down && docker-compose build --no-cache && docker-compose up -d

# Check container health
docker-compose ps
docker inspect forexswing-ai-bot | grep Health

# Access container shell
docker exec -it forexswing-ai-bot /bin/bash

# View real-time logs
docker-compose logs -f --tail=100
```

---

## 📱 Monitoring

### Web Interface
```
http://your-vm-ip:8082
```

### Terminal Dashboard
```bash
python src/monitoring/dashboard.py http://your-vm-ip:8082
```

### Logs
```bash
# Docker logs
docker-compose logs -f

# System logs
sudo journalctl -u forexbot -f

# Application logs
tail -f data/logs/data_collector.log
```

---

## 🎓 Learning Resources

- **Full Deployment Guide**: [DEPLOYMENT.md](DEPLOYMENT.md)
- **VM Setup Summary**: [VM_DEPLOYMENT_SUMMARY.md](VM_DEPLOYMENT_SUMMARY.md)
- **Project Overview**: [README.md](README.md)
- **Architecture**: Check source code comments

---

## ✅ Health Check Checklist

```bash
# 1. Containers running?
docker-compose ps
# Expected: forexbot, data-collector (Up)

# 2. API responding?
curl http://localhost:8082/api/status
# Expected: {"status": "operational", ...}

# 3. Data collected?
ls data/MarketData/*.csv
# Expected: Multiple CSV files for each pair

# 4. Model loaded?
curl "http://localhost:8082/api/analyze?pair=EUR/USD" | grep -i "models_used"
# Expected: "lstm": "active"

# 5. News working?
ls data/News/*.json
# Expected: JSON files with recent dates

# 6. Disk space?
df -h
# Expected: >5GB free

# 7. Memory usage?
docker stats --no-stream
# Expected: <2GB per container
```

---

## 🎯 Common Use Cases

### Get Trading Signal
```python
import requests
response = requests.get('http://your-vm-ip:8082/api/analyze?pair=EUR/USD')
data = response.json()
print(f"Action: {data['action']}, Confidence: {data['confidence']:.1%}")
```

### Batch Analysis
```bash
for pair in EUR/USD GBP/USD USD/JPY; do
  echo "=== $pair ==="
  curl -s "http://localhost:8082/api/analyze?pair=$pair" | python -m json.tool
done
```

### Continuous Monitoring
```bash
watch -n 60 'curl -s "http://localhost:8082/api/analyze?pair=EUR/USD" | python -m json.tool'
```

---

## 📞 Support

- **Documentation**: See [DEPLOYMENT.md](DEPLOYMENT.md)
- **Issues**: Check logs in `data/logs/`
- **Quick Help**: Run `./quick_start.sh`

---

**Keep this reference handy! Bookmark or print for quick access.**

**Questions?** See [VM_DEPLOYMENT_SUMMARY.md](VM_DEPLOYMENT_SUMMARY.md) for detailed explanations.
