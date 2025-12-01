# ForexSwing AI 2025 - Native Ubuntu Deployment (No Docker)

## When to Use This Method

Use native deployment if you:
- Don't want to use Docker
- Want direct Python installation
- Need maximum control
- Have minimal resources

**Pros:** Simple, direct access, no Docker overhead
**Cons:** Less isolated, manual dependency management

---

## Quick Deployment (Ubuntu 20.04/22.04)

### Step 1: System Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11 (if not available, use 3.10+)
sudo apt install -y python3 python3-pip python3-venv

# Install system dependencies
sudo apt install -y \
    build-essential \
    curl \
    git \
    wget \
    libgomp1 \
    libopenblas-dev \
    python3-dev

# Optional: Install Python 3.11 from deadsnakes PPA
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt install python3.11 python3.11-venv python3.11-dev -y
```

### Step 2: Create Application User

```bash
# Create dedicated user for bot
sudo useradd -m -s /bin/bash forexbot

# Set password
sudo passwd forexbot

# Add to necessary groups
sudo usermod -aG sudo forexbot

# Switch to forexbot user
sudo su - forexbot
```

### Step 3: Setup Application

```bash
# Create application directory
mkdir -p /home/forexbot/forexswing-ai-2025
cd /home/forexbot/forexswing-ai-2025
```

**Copy files (from another terminal/machine):**

```bash
# From your Windows machine or Proxmox
scp -r "G:\Trading bot stuff\forexswing-ai-2025\*" forexbot@[server-ip]:/home/forexbot/forexswing-ai-2025/
```

**Back on the server:**

```bash
cd /home/forexbot/forexswing-ai-2025

# Create Python virtual environment
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_app.txt

# Install additional production dependencies
pip install schedule python-dotenv yfinance alpha_vantage google-generativeai
```

### Step 4: Configure Environment

```bash
# Create .env file
cp .env.example .env
nano .env
```

Add your API keys:
```bash
ALPHA_VANTAGE_KEY=your_alpha_vantage_key
GOOGLE_API_KEY=your_gemini_api_key
NEWS_API_KEY=your_news_api_key (optional)
```

### Step 5: Create Directories

```bash
# Create necessary directories
mkdir -p data/models data/MarketData data/News data/logs
```

### Step 6: Initial Data Collection

```bash
# Activate venv if not already
source venv/bin/activate

# Run data collector (takes 15-20 minutes)
python src/data/market_data_collector.py --once
```

---

## Create Systemd Services

### Service 1: API Server

```bash
# Create service file
sudo nano /etc/systemd/system/forexbot-api.service
```

```ini
[Unit]
Description=ForexSwing AI 2025 - API Service
After=network.target

[Service]
Type=simple
User=forexbot
Group=forexbot
WorkingDirectory=/home/forexbot/forexswing-ai-2025
Environment="PATH=/home/forexbot/forexswing-ai-2025/venv/bin"
ExecStart=/home/forexbot/forexswing-ai-2025/venv/bin/python companion_api_service.py 8082
Restart=always
RestartSec=10
StandardOutput=append:/home/forexbot/forexswing-ai-2025/data/logs/api.log
StandardError=append:/home/forexbot/forexswing-ai-2025/data/logs/api-error.log

[Install]
WantedBy=multi-user.target
```

### Service 2: Data Collector

```bash
# Create service file
sudo nano /etc/systemd/system/forexbot-collector.service
```

```ini
[Unit]
Description=ForexSwing AI 2025 - Data Collector
After=network.target

[Service]
Type=simple
User=forexbot
Group=forexbot
WorkingDirectory=/home/forexbot/forexswing-ai-2025
Environment="PATH=/home/forexbot/forexswing-ai-2025/venv/bin"
ExecStart=/home/forexbot/forexswing-ai-2025/venv/bin/python src/data/market_data_collector.py
Restart=always
RestartSec=10
StandardOutput=append:/home/forexbot/forexswing-ai-2025/data/logs/collector.log
StandardError=append:/home/forexbot/forexswing-ai-2025/data/logs/collector-error.log

[Install]
WantedBy=multi-user.target
```

### Enable and Start Services

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable services
sudo systemctl enable forexbot-api
sudo systemctl enable forexbot-collector

# Start services
sudo systemctl start forexbot-api
sudo systemctl start forexbot-collector

# Check status
sudo systemctl status forexbot-api
sudo systemctl status forexbot-collector
```

---

## Service Management

### View Logs

```bash
# API logs
sudo journalctl -u forexbot-api -f

# Collector logs
sudo journalctl -u forexbot-collector -f

# Application logs
tail -f /home/forexbot/forexswing-ai-2025/data/logs/api.log
tail -f /home/forexbot/forexswing-ai-2025/data/logs/collector.log
```

### Control Services

```bash
# Start
sudo systemctl start forexbot-api
sudo systemctl start forexbot-collector

# Stop
sudo systemctl stop forexbot-api
sudo systemctl stop forexbot-collector

# Restart
sudo systemctl restart forexbot-api
sudo systemctl restart forexbot-collector

# Check status
sudo systemctl status forexbot-api
sudo systemctl status forexbot-collector
```

---

## Manual Operations

### Run Data Collector Manually

```bash
sudo su - forexbot
cd /home/forexbot/forexswing-ai-2025
source venv/bin/activate
python src/data/market_data_collector.py --once
```

### Train Model Manually

```bash
sudo su - forexbot
cd /home/forexbot/forexswing-ai-2025
source venv/bin/activate
python src/training/enhanced_model_trainer.py
```

### Test API

```bash
# From server
curl http://localhost:8082/api/status

# Get analysis
curl "http://localhost:8082/api/analyze?pair=EUR/USD"
```

---

## Monitoring Dashboard

### Install Dashboard Dependencies

```bash
source venv/bin/activate
pip install requests
```

### Run Dashboard

```bash
# In screen session (stays running after logout)
screen -S dashboard
source venv/bin/activate
python src/monitoring/dashboard.py http://localhost:8082

# Detach: Ctrl+A, then D
# Reattach: screen -r dashboard
```

---

## Automated Tasks with Cron

### Alternative to Systemd Collector Service

```bash
# Edit crontab for forexbot user
crontab -e
```

Add:
```bash
# Full data collection daily at midnight
0 0 * * * cd /home/forexbot/forexswing-ai-2025 && /home/forexbot/forexswing-ai-2025/venv/bin/python src/data/market_data_collector.py --once >> data/logs/cron-collector.log 2>&1

# Quick news update every 4 hours
0 */4 * * * cd /home/forexbot/forexswing-ai-2025 && /home/forexbot/forexswing-ai-2025/venv/bin/python -c "from src.data.market_data_collector import EnhancedMarketDataCollector; collector = EnhancedMarketDataCollector(); collector.run_quick_update()" >> data/logs/cron-news.log 2>&1
```

---

## Firewall Setup

```bash
# Install UFW
sudo apt install ufw

# Allow SSH
sudo ufw allow 22/tcp

# Allow API port
sudo ufw allow 8082/tcp

# Enable firewall
sudo ufw enable

# Check status
sudo ufw status
```

---

## Performance Tuning

### Optimize Python Performance

```bash
# Install optimized libraries
source venv/bin/activate
pip install numpy --upgrade --force-reinstall --no-binary numpy  # Compile with optimizations
```

### System Limits

```bash
# Increase file descriptors for user
sudo nano /etc/security/limits.conf
```

Add:
```
forexbot soft nofile 65536
forexbot hard nofile 65536
```

---

## Backup Strategy

### Create Backup Script

```bash
nano /home/forexbot/backup.sh
```

```bash
#!/bin/bash
BACKUP_DIR="/home/forexbot/backups"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup data directory
tar -czf $BACKUP_DIR/data-$DATE.tar.gz \
    /home/forexbot/forexswing-ai-2025/data/

# Backup .env
cp /home/forexbot/forexswing-ai-2025/.env \
   $BACKUP_DIR/env-$DATE.bak

# Keep only last 7 days
find $BACKUP_DIR -type f -mtime +7 -delete

echo "Backup completed: $DATE"
```

```bash
chmod +x /home/forexbot/backup.sh

# Schedule daily backup at 3 AM
crontab -e
```

Add:
```
0 3 * * * /home/forexbot/backup.sh >> /home/forexbot/backup.log 2>&1
```

---

## Update Procedure

```bash
# 1. Stop services
sudo systemctl stop forexbot-api forexbot-collector

# 2. Backup current version
cd /home/forexbot
tar -czf forexswing-backup-$(date +%Y%m%d).tar.gz forexswing-ai-2025/

# 3. Pull updates (if using git)
cd forexswing-ai-2025
git pull

# Or copy new files
# scp -r new-files/* forexbot@server:/home/forexbot/forexswing-ai-2025/

# 4. Update dependencies
source venv/bin/activate
pip install -r requirements.txt --upgrade

# 5. Restart services
sudo systemctl start forexbot-api forexbot-collector

# 6. Check status
sudo systemctl status forexbot-api forexbot-collector
```

---

## Troubleshooting

### Service Won't Start

```bash
# Check logs
sudo journalctl -u forexbot-api -n 50
sudo journalctl -u forexbot-collector -n 50

# Check permissions
ls -la /home/forexbot/forexswing-ai-2025

# Test manually
sudo su - forexbot
cd forexswing-ai-2025
source venv/bin/activate
python companion_api_service.py 8082
```

### Python Module Not Found

```bash
# Ensure venv is activated
source venv/bin/activate

# Reinstall requirements
pip install -r requirements.txt
pip install -r requirements_app.txt
```

### Permission Errors

```bash
# Fix ownership
sudo chown -R forexbot:forexbot /home/forexbot/forexswing-ai-2025

# Fix permissions
chmod -R 755 /home/forexbot/forexswing-ai-2025
chmod 600 /home/forexbot/forexswing-ai-2025/.env
```

### Port Already in Use

```bash
# Find what's using port 8082
sudo lsof -i :8082

# Or
sudo netstat -tulpn | grep 8082

# Kill the process
sudo kill -9 [PID]
```

---

## Resource Monitoring

### System Resources

```bash
# CPU and memory
htop

# Disk usage
df -h
du -sh /home/forexbot/forexswing-ai-2025/*

# Network
sudo iftop
```

### Application Monitoring

```bash
# Process status
ps aux | grep python

# Service resources
systemctl status forexbot-api
systemctl status forexbot-collector
```

---

## Security Best Practices

1. **Use dedicated user** (forexbot)
2. **Virtual environment** for Python isolation
3. **Firewall** enabled (UFW)
4. **Regular updates**
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```
5. **Secure .env file**
   ```bash
   chmod 600 .env
   ```
6. **Log rotation**
   ```bash
   sudo nano /etc/logrotate.d/forexbot
   ```
   ```
   /home/forexbot/forexswing-ai-2025/data/logs/*.log {
       daily
       rotate 7
       compress
       missingok
       notifempty
   }
   ```

---

## Quick Reference

```bash
# START/STOP
sudo systemctl start forexbot-api forexbot-collector
sudo systemctl stop forexbot-api forexbot-collector
sudo systemctl restart forexbot-api forexbot-collector

# STATUS
sudo systemctl status forexbot-api
sudo systemctl status forexbot-collector

# LOGS
sudo journalctl -u forexbot-api -f
tail -f data/logs/api.log

# TEST
curl http://localhost:8082/api/status
curl "http://localhost:8082/api/analyze?pair=EUR/USD"

# MANUAL OPERATIONS
sudo su - forexbot
cd forexswing-ai-2025
source venv/bin/activate
python src/data/market_data_collector.py --once
```

---

## Advantages vs Docker

| Feature | Native | Docker |
|---------|--------|--------|
| **RAM Usage** | Lower | Higher |
| **Complexity** | Simple | Moderate |
| **Isolation** | Less | More |
| **Updates** | Manual | Container rebuild |
| **Debugging** | Easier | Requires exec |
| **Performance** | Slightly better | Good |

---

## Next Steps

1. ✅ Install Python and dependencies
2. ✅ Create forexbot user
3. ✅ Copy application files
4. ✅ Setup virtual environment
5. ✅ Configure .env
6. ✅ Create systemd services
7. ✅ Start services
8. ✅ Setup backups

**Your bot is running natively on Ubuntu!**
