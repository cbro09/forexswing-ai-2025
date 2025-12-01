# ForexSwing AI 2025 - Proxmox Deployment Guide

## 🎯 You Have Proxmox! Perfect Choice!

Proxmox is **IDEAL** for running this trading bot. You have **THREE excellent deployment options**:

---

## 📋 Deployment Options Comparison

| Option | RAM | Disk | Boot Time | Complexity | Recommended For |
|--------|-----|------|-----------|------------|-----------------|
| **1. LXC Container** | 2GB | 10GB | 3 sec | Easy | **MOST USERS** ⭐ |
| **2. Proxmox VM** | 4GB | 20GB | 30 sec | Easy | Full isolation |
| **3. Native Ubuntu** | 2GB | 15GB | N/A | Medium | No Docker |

---

## ⭐ **RECOMMENDED: LXC Container (Fastest & Most Efficient)**

### Why LXC?
✅ **Lightweight** - Uses 1/4 the RAM of a VM
✅ **Fast** - Boots in 3 seconds
✅ **Efficient** - Near-native performance
✅ **Easy backups** - Built-in Proxmox snapshots
✅ **Perfect for Docker** - Runs containers efficiently

### Quick Start (5 Minutes)

```bash
# 1. Create LXC container in Proxmox Web UI
#    - Template: ubuntu-22.04-standard
#    - CPU: 2 cores
#    - RAM: 2GB (4GB recommended)
#    - Disk: 20GB
#    - Network: DHCP or static IP

# 2. Enable Docker features (from Proxmox shell)
pct set 100 -features nesting=1,keyctl=1  # Replace 100 with your CT ID
pct start 100

# 3. Enter container
pct enter 100

# 4. Install Docker
curl -fsSL https://get.docker.com | sh

# 5. Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# 6. Copy your files (from Windows to Proxmox to container)
# From Proxmox host:
pct push 100 /tmp/forexbot /opt/forexswing-ai-2025 -r

# 7. Deploy (inside container)
cd /opt/forexswing-ai-2025
cp .env.example .env
nano .env  # Add API keys
docker-compose build
docker-compose up -d

# 8. Collect data and start
docker-compose run data-collector python src/data/market_data_collector.py --once
```

**Done! Access at `http://[container-ip]:8082`**

**📖 Full Guide:** [deploy/PROXMOX_LXC_DEPLOYMENT.md](deploy/PROXMOX_LXC_DEPLOYMENT.md)

---

## 🖥️ **Option 2: Traditional VM (More Isolation)**

### When to Use
- Need full OS isolation
- Want to test different distributions
- Prefer traditional VM management

### Quick Start (10 Minutes)

```bash
# 1. Create VM in Proxmox Web UI
#    - OS: Ubuntu 22.04 Server ISO
#    - CPU: 2-4 cores
#    - RAM: 4GB
#    - Disk: 20GB

# 2. Install Ubuntu via console

# 3. SSH into VM
ssh admin@[vm-ip]

# 4. Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# 5. Copy files and deploy
scp -r "forexswing-ai-2025/*" admin@[vm-ip]:/opt/forexswing-ai-2025/
cd /opt/forexswing-ai-2025
docker-compose build
docker-compose up -d
```

**📖 Full Guide:** [deploy/PROXMOX_VM_DEPLOYMENT.md](deploy/PROXMOX_VM_DEPLOYMENT.md)

---

## 🐍 **Option 3: Native Python (No Docker)**

### When to Use
- Don't want Docker overhead
- Need maximum control
- Minimal resources

### Quick Start (15 Minutes)

```bash
# 1. Create LXC or VM with Ubuntu

# 2. Install Python
apt install python3.11 python3.11-venv -y

# 3. Create user and setup
useradd -m forexbot
su - forexbot
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. Create systemd services
sudo cp deploy/forexbot-api.service /etc/systemd/system/
sudo systemctl enable forexbot-api
sudo systemctl start forexbot-api
```

**📖 Full Guide:** [deploy/NATIVE_UBUNTU_DEPLOYMENT.md](deploy/NATIVE_UBUNTU_DEPLOYMENT.md)

---

## 📦 **Transferring Files from Windows to Proxmox**

### Method 1: SCP (Recommended)

```powershell
# From Windows PowerShell
cd "G:\Trading bot stuff\forexswing-ai-2025"

# Copy to Proxmox host
scp -r * root@proxmox-ip:/tmp/forexbot/

# Then from Proxmox shell, push to LXC
pct push 100 /tmp/forexbot /opt/forexswing-ai-2025 -r

# Or copy to VM
scp -r /tmp/forexbot/* admin@vm-ip:/opt/forexswing-ai-2025/
```

### Method 2: Create Deployment Package (Easiest)

```batch
# On Windows
setup_windows.bat
# Select: 4) Prepare for VM deployment

# This creates deployment_package folder
# Then copy deployment_package to Proxmox
```

### Method 3: WinSCP GUI

1. Download WinSCP: https://winscp.net/
2. Connect to Proxmox: `root@proxmox-ip`
3. Drag & drop `forexswing-ai-2025` folder to `/tmp/`
4. From Proxmox shell: `pct push 100 /tmp/forexswing-ai-2025 /opt/forexswing-ai-2025 -r`

---

## 🔧 **Proxmox-Specific Features**

### Backups

**Automated Daily Backups:**

```bash
# From Proxmox Web UI:
# 1. Datacenter → Backup → Add
# 2. Schedule: Daily at 2 AM
# 3. Selection: Your container/VM
# 4. Retention: Keep 7 backups

# Or from Proxmox shell:
cat > /etc/cron.d/forexbot-backup <<EOF
0 2 * * * root vzdump 100 --compress zstd --mode snapshot --storage local
EOF
```

### Snapshots

```bash
# LXC snapshot
pct snapshot 100 before-update

# VM snapshot (from Proxmox UI)
# Select VM → Snapshots → Take Snapshot

# Rollback if something breaks
pct rollback 100 before-update
```

### Resource Monitoring

**From Proxmox Web UI:**
- Select container/VM
- Summary tab shows real-time graphs
- CPU, RAM, Network, Disk I/O

**From command line:**

```bash
# LXC resources
pct status 100 --verbose

# VM resources
qm status 100 --verbose
```

### Cloning for Testing

```bash
# Clone LXC
pct clone 100 101 --full --hostname forexbot-test

# Clone VM
qm clone 100 101 --full --name forexbot-test
```

---

## 🌐 **Networking on Proxmox**

### Access Bot from LAN

```bash
# Bot automatically accessible at:
http://[container-ip]:8082

# Find container IP:
pct exec 100 -- ip addr show eth0
```

### Port Forwarding (Access from Internet)

**From Proxmox host:**

```bash
# Forward port 8082 to container
iptables -t nat -A PREROUTING -p tcp --dport 8082 -j DNAT --to [container-ip]:8082
iptables -t nat -A POSTROUTING -s [container-ip] -j MASQUERADE

# Make persistent
apt install iptables-persistent
netfilter-persistent save
```

### Reverse Proxy (Optional, for HTTPS)

**Install nginx on Proxmox or separate container:**

```bash
apt install nginx

# Create config
cat > /etc/nginx/sites-available/forexbot <<EOF
server {
    listen 80;
    server_name forexbot.yourdomain.com;

    location / {
        proxy_pass http://[container-ip]:8082;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
}
EOF

ln -s /etc/nginx/sites-available/forexbot /etc/nginx/sites-enabled/
systemctl restart nginx
```

---

## 💰 **Cost Breakdown**

### Hardware (One-time)
- **Proxmox Server**: Existing (FREE for you!) ✅
- **Storage**: Uses existing Proxmox storage

### Operating Costs
- **Electricity**: ~$5-10/month (depends on server)
- **API Calls**: FREE (Alpha Vantage, Gemini)

### **Total: ~$0-10/month** (vs $25-35/month for cloud VMs)

---

## ⚡ **Performance Tips for Proxmox**

### LXC Container Optimization

```bash
# Allocate more CPU cores
pct set 100 -cores 4

# Increase RAM
pct set 100 -memory 4096

# Enable swap
pct set 100 -swap 1024
```

### VM Optimization

```bash
# Use VirtIO for best performance
# In Proxmox UI: Hardware → Hard Disk → Bus: VirtIO
# Network Device → Model: VirtIO

# Enable qemu-guest-agent in VM
apt install qemu-guest-agent
systemctl enable qemu-guest-agent

# In Proxmox
qm set 100 -agent enabled=1
```

### Storage Optimization

```bash
# Use local SSD storage if available
# In Proxmox UI: When creating container/VM
# Storage: local-lvm (SSD) vs local-zfs (HDD)
```

---

## 🔒 **Security Best Practices**

1. **Firewall in Container/VM**
   ```bash
   apt install ufw
   ufw allow 8082/tcp
   ufw enable
   ```

2. **Proxmox Firewall**
   ```bash
   # From Proxmox UI:
   # Datacenter → Firewall → Enable
   # VM/CT → Firewall → Add rules
   ```

3. **API Authentication** (Optional)
   - Add nginx reverse proxy with basic auth
   - Use API keys
   - Implement rate limiting

4. **Regular Updates**
   ```bash
   # Inside container/VM
   apt update && apt upgrade -y
   docker-compose pull
   docker-compose up -d
   ```

---

## 🆘 **Troubleshooting Proxmox-Specific Issues**

### Container Won't Start

```bash
# Check status
pct status 100

# Check logs
pct log 100

# Force start
pct start 100 --force
```

### Docker Not Working in LXC

```bash
# Ensure features are enabled
pct set 100 -features nesting=1,keyctl=1

# Restart container
pct reboot 100

# Inside container, check Docker
systemctl status docker
```

### Network Issues

```bash
# Check Proxmox bridge
brctl show

# Check container network
pct exec 100 -- ip addr

# Restart networking
pct reboot 100
```

### Out of Memory

```bash
# Check current allocation
pct config 100 | grep memory

# Increase memory
pct set 100 -memory 4096 -swap 1024

# Restart container
pct reboot 100
```

---

## 📊 **Monitoring Dashboard**

### Built-in Proxmox Monitoring
- Web UI: Select container → Summary
- Real-time graphs for CPU, RAM, Network, Disk

### Custom Monitoring

```bash
# Inside container
python src/monitoring/dashboard.py http://localhost:8082

# Or from Proxmox host
curl "http://[container-ip]:8082/api/status"
```

---

## 🎯 **Recommended Setup: LXC + Docker**

**Best of both worlds:**
- ✅ LXC container (lightweight, fast)
- ✅ Docker inside (isolated apps, easy management)
- ✅ Proxmox backups/snapshots (easy disaster recovery)

**Resource Usage:**
- RAM: ~1GB (container) + ~500MB (Docker) = **1.5GB total**
- Disk: ~10GB (very efficient)
- CPU: Minimal when idle, <10% under load

---

## 📚 **Complete Documentation**

1. **[PROXMOX_LXC_DEPLOYMENT.md](deploy/PROXMOX_LXC_DEPLOYMENT.md)** ⭐ **START HERE**
   - Detailed LXC container setup
   - Docker installation
   - Port forwarding
   - Backups and snapshots

2. **[PROXMOX_VM_DEPLOYMENT.md](deploy/PROXMOX_VM_DEPLOYMENT.md)**
   - Traditional VM setup
   - Ubuntu installation
   - Resource management

3. **[NATIVE_UBUNTU_DEPLOYMENT.md](deploy/NATIVE_UBUNTU_DEPLOYMENT.md)**
   - No Docker deployment
   - Systemd services
   - Manual management

4. **[VM_DEPLOYMENT_SUMMARY.md](VM_DEPLOYMENT_SUMMARY.md)**
   - General VM deployment overview
   - Works with Proxmox too

5. **[DEPLOYMENT.md](DEPLOYMENT.md)**
   - Complete deployment guide
   - All platforms

6. **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)**
   - Command cheat sheet

---

## 🚀 **Quick Start Commands**

### Create LXC Container (Fastest)

```bash
# 1. Proxmox Web UI → Create CT
# 2. From Proxmox shell:
pct set 100 -features nesting=1,keyctl=1
pct start 100
pct enter 100

# 3. Inside container:
curl -fsSL https://get.docker.com | sh
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# 4. Deploy:
cd /opt/forexswing-ai-2025
docker-compose build && docker-compose up -d
```

**That's it! Bot running in 5 minutes.** ✅

---

## ✅ **Success Checklist**

After deployment, verify:

- [ ] Container/VM is running in Proxmox
- [ ] Can SSH/console into container/VM
- [ ] Docker is installed and working: `docker --version`
- [ ] Bot API responds: `curl http://localhost:8082/api/status`
- [ ] Data directory populated: `ls data/MarketData/`
- [ ] Services auto-start on reboot
- [ ] Backups configured in Proxmox
- [ ] Can access from LAN: `http://[container-ip]:8082`

---

## 🎉 **Your Proxmox Advantage**

You already have Proxmox, which means:
- ✅ **No cloud costs** - Run locally for free
- ✅ **Full control** - Complete server access
- ✅ **Easy backups** - Built-in snapshot system
- ✅ **Flexibility** - Can run multiple containers/VMs
- ✅ **Performance** - Direct hardware access
- ✅ **Privacy** - All data stays local

**This is actually BETTER than deploying to a cloud VM!**

---

## 🎯 **Next Steps**

1. **Choose deployment method** (LXC recommended)
2. **Follow the appropriate guide**
   - [PROXMOX_LXC_DEPLOYMENT.md](deploy/PROXMOX_LXC_DEPLOYMENT.md) ⭐
   - [PROXMOX_VM_DEPLOYMENT.md](deploy/PROXMOX_VM_DEPLOYMENT.md)
   - [NATIVE_UBUNTU_DEPLOYMENT.md](deploy/NATIVE_UBUNTU_DEPLOYMENT.md)
3. **Get API keys** (5 min, both free)
4. **Deploy and test** (5-15 min)
5. **Start collecting data**
6. **Monitor and enjoy!**

---

**Your ForexSwing AI bot is ready to run 24/7 on your Proxmox server!** 🚀

Need help? Check the detailed guides above or use the interactive menu: `./quick_start.sh`
