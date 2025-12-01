# ForexSwing AI 2025 - Proxmox LXC Container Deployment

## Why LXC Container? (RECOMMENDED)

✅ **Lightweight** - Uses ~500MB RAM vs 2GB for VM
✅ **Fast** - Near-native performance
✅ **Easy backups** - Proxmox snapshot/backup built-in
✅ **Low overhead** - No full OS virtualization
✅ **Perfect for Docker** - Runs Docker containers efficiently

---

## Quick Deployment (5 Minutes)

### Step 1: Create LXC Container in Proxmox

**From Proxmox Web UI:**

1. Click **"Create CT"** (top right)
2. Configure:
   ```
   General:
   - Hostname: forexbot
   - Unprivileged container: ✓ (checked)
   - Password: [set secure password]

   Template:
   - Storage: local
   - Template: ubuntu-22.04-standard (download if needed)

   Root Disk:
   - Disk size: 20 GB (minimum 15GB)

   CPU:
   - Cores: 2 (minimum)

   Memory:
   - Memory (MiB): 2048 (2GB minimum, 4GB recommended)
   - Swap (MiB): 512

   Network:
   - Bridge: vmbr0
   - IPv4: DHCP or static (note the IP!)
   - IPv6: DHCP or none

   DNS:
   - Use host settings: ✓
   ```

3. Click **"Finish"** but **DON'T START YET**

### Step 2: Enable Features for Docker

**In Proxmox shell (or SSH to Proxmox host):**

```bash
# Replace 100 with your actual CT ID
CT_ID=100

# Enable required features for Docker
pct set $CT_ID -features nesting=1,keyctl=1

# Optional: For better performance
pct set $CT_ID -onboot 1

# Start the container
pct start $CT_ID
```

### Step 3: Enter Container and Setup

```bash
# Enter container console
pct enter $CT_ID

# OR SSH into container
ssh root@[container-ip]
```

**Inside the container:**

```bash
# Update system
apt update && apt upgrade -y

# Install required packages
apt install -y curl git wget ca-certificates gnupg lsb-release

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
rm get-docker.sh

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Verify installations
docker --version
docker-compose --version
```

### Step 4: Deploy ForexSwing AI

```bash
# Create application directory
mkdir -p /opt/forexswing-ai-2025
cd /opt/forexswing-ai-2025

# Copy your files here (from Windows to Proxmox host, then to container)
# Or use git clone if you have a repository
```

**From your Windows machine, copy files to Proxmox:**

```powershell
# Method 1: Copy to Proxmox host first
scp -r "G:\Trading bot stuff\forexswing-ai-2025\*" root@proxmox-ip:/tmp/forexbot/

# Then from Proxmox host to container
pct push 100 /tmp/forexbot /opt/forexswing-ai-2025 -r
```

**Or use WinSCP/FileZilla to copy files**

**Back in the container:**

```bash
cd /opt/forexswing-ai-2025

# Make scripts executable
chmod +x deploy/setup_vm.sh quick_start.sh

# Setup environment
cp .env.example .env
nano .env
# Add your API keys:
# ALPHA_VANTAGE_KEY=your_key
# GOOGLE_API_KEY=your_gemini_key

# Build and start services
docker-compose build
docker-compose up -d

# Collect initial data (optional, takes 15-20 min)
docker-compose run --rm data-collector python src/data/market_data_collector.py --once

# Check status
docker-compose ps
curl http://localhost:8082/api/status
```

### Step 5: Setup Auto-Start (Optional)

```bash
# Install systemd service
cp deploy/forexbot.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable forexbot
systemctl start forexbot

# Check status
systemctl status forexbot
```

---

## Access Your Bot

### From Proxmox Network:
```
http://[container-ip]:8082
```

### From Anywhere (Setup Port Forward in Proxmox):

**In Proxmox host shell:**

```bash
# Add iptables rule for port forwarding
iptables -t nat -A PREROUTING -i vmbr0 -p tcp --dport 8082 -j DNAT --to [container-ip]:8082
iptables -t nat -A POSTROUTING -s [container-ip] -o vmbr0 -j MASQUERADE

# Make persistent (create script)
cat > /etc/network/if-up.d/forexbot-forward <<'EOF'
#!/bin/bash
iptables -t nat -A PREROUTING -i vmbr0 -p tcp --dport 8082 -j DNAT --to [container-ip]:8082
iptables -t nat -A POSTROUTING -s [container-ip] -o vmbr0 -j MASQUERADE
EOF

chmod +x /etc/network/if-up.d/forexbot-forward
```

---

## LXC Container Management

### Common Commands

```bash
# Start container
pct start 100

# Stop container
pct stop 100

# Restart container
pct reboot 100

# Enter console
pct enter 100

# Check status
pct status 100

# Resource usage
pct status 100 --verbose
```

### Backups

**From Proxmox Web UI:**
1. Select container
2. Click "Backup" tab
3. Click "Backup now"
4. Choose storage location

**From Proxmox shell:**

```bash
# Manual backup
vzdump 100 --compress zstd --mode snapshot --storage local

# Automated daily backups at 2 AM
cat > /etc/cron.d/forexbot-backup <<EOF
0 2 * * * root vzdump 100 --compress zstd --mode snapshot --storage local
EOF
```

### Snapshots

```bash
# Create snapshot
pct snapshot 100 before-update

# List snapshots
pct listsnapshot 100

# Restore snapshot
pct rollback 100 before-update

# Delete snapshot
pct delsnapshot 100 before-update
```

---

## Resource Monitoring

### From Proxmox Web UI:
- Select container → Summary tab
- View CPU, RAM, Network usage graphs

### From Container:

```bash
# Real-time resource usage
docker stats

# Container logs
docker-compose logs -f

# Check disk usage
df -h
```

---

## Performance Tuning for LXC

### Allocate More Resources

```bash
# From Proxmox host
pct set 100 -cores 4        # Increase CPU cores
pct set 100 -memory 4096    # Increase RAM to 4GB
pct set 100 -swap 1024      # Increase swap
```

### Optimize I/O

```bash
# Enable I/O throttling limits (prevent resource hogging)
pct set 100 -mp0 /opt/forexswing-ai-2025,mp=/opt/data,backup=1
```

---

## Troubleshooting

### Docker Not Starting in LXC

```bash
# Ensure nesting is enabled
pct set 100 -features nesting=1,keyctl=1

# Restart container
pct reboot 100
```

### Container Out of Memory

```bash
# Increase memory from Proxmox host
pct set 100 -memory 4096 -swap 1024

# Restart container
pct reboot 100
```

### Network Issues

```bash
# Inside container, check network
ip addr
ping google.com

# From Proxmox host, check bridge
brctl show
```

### Permission Errors with Docker

```bash
# Inside container
systemctl status docker
journalctl -u docker -f
```

---

## Migration & Scaling

### Clone Container

**From Proxmox Web UI:**
1. Right-click container → Clone
2. Set new CT ID and name
3. Full clone recommended

### Move to Different Node (Proxmox Cluster)

```bash
# From Proxmox shell
pct migrate 100 [target-node] --online
```

---

## Security Best Practices

1. **Firewall Rules**
   ```bash
   # Inside container
   apt install ufw
   ufw allow 8082/tcp
   ufw enable
   ```

2. **Update Regularly**
   ```bash
   # Inside container
   apt update && apt upgrade -y
   docker-compose pull
   docker-compose up -d
   ```

3. **Secure API Access**
   - Use reverse proxy (nginx/traefik)
   - Add authentication
   - Use HTTPS

---

## Advantages of LXC vs Traditional VM

| Feature | LXC Container | Traditional VM |
|---------|---------------|----------------|
| **RAM Usage** | ~500MB | ~2GB |
| **Boot Time** | <5 seconds | 30-60 seconds |
| **Performance** | Near-native | Slight overhead |
| **Disk Space** | ~2-3GB | ~10-20GB |
| **Snapshots** | Instant | Minutes |
| **Backups** | Fast | Slower |

---

## Quick Reference Commands

```bash
# PROXMOX HOST
pct list                          # List all containers
pct start 100                     # Start container
pct stop 100                      # Stop container
pct enter 100                     # Enter console
pct snapshot 100 snapshot-name    # Create snapshot
vzdump 100                        # Backup container

# INSIDE CONTAINER
docker-compose up -d              # Start services
docker-compose down               # Stop services
docker-compose logs -f            # View logs
docker-compose ps                 # Check status
./quick_start.sh                  # Interactive menu
```

---

## Cost

**FREE!**
- Uses existing Proxmox server
- No cloud hosting fees
- Only needs internet for API calls (free tier)

---

## Next Steps

1. ✅ Create LXC container in Proxmox
2. ✅ Enable Docker features
3. ✅ Copy ForexSwing AI files
4. ✅ Configure API keys
5. ✅ Start services
6. ✅ Collect initial data
7. ✅ Set up backups

**You're done! Bot running 24/7 on Proxmox LXC.**

---

## Support

- **Container issues**: Check Proxmox logs
- **Docker issues**: `docker-compose logs -f`
- **Bot issues**: See main [DEPLOYMENT.md](../DEPLOYMENT.md)

**Proxmox LXC is the BEST option for your use case!** 🚀
