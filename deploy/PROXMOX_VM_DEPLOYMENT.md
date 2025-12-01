# ForexSwing AI 2025 - Proxmox VM Deployment

## When to Use VM Instead of LXC

Use a traditional VM if you:
- Need full OS isolation
- Want to test different Linux distributions
- Need kernel-level modifications
- Prefer more separation from host

**For most users, LXC is better (see PROXMOX_LXC_DEPLOYMENT.md)**

---

## Quick VM Deployment

### Step 1: Download Ubuntu ISO

**From Proxmox Web UI:**

1. Select your Proxmox node
2. Click **"local (pve)"** → **"ISO Images"**
3. Click **"Download from URL"**
4. URL: `https://releases.ubuntu.com/22.04/ubuntu-22.04.3-live-server-amd64.iso`
5. Wait for download to complete

### Step 2: Create VM

**From Proxmox Web UI:**

1. Click **"Create VM"** (top right)
2. Configure:

   ```
   General:
   - VM ID: 100 (or any available)
   - Name: forexbot

   OS:
   - ISO image: ubuntu-22.04.3-live-server-amd64.iso
   - Guest OS: Linux 5.x - 2.6 Kernel

   System:
   - Default settings (OVMF not needed unless UEFI required)

   Disks:
   - Bus/Device: SCSI
   - Storage: local-lvm
   - Disk size: 20 GiB minimum
   - Cache: Write back
   - Discard: ✓ (if SSD)

   CPU:
   - Cores: 2 minimum (4 recommended)
   - Type: host (for best performance)

   Memory:
   - Memory (MiB): 4096 (4GB minimum)

   Network:
   - Bridge: vmbr0
   - Model: VirtIO (paravirtualized)
   ```

3. Click **"Finish"**
4. **Start VM** and open **Console**

### Step 3: Install Ubuntu

**Follow Ubuntu installer:**

1. Choose language → English
2. Update installer → Skip (Continue without updating)
3. Keyboard → English (US)
4. Network → Use DHCP (note the IP address!)
5. Proxy → Leave blank
6. Mirror → Default
7. Storage → Use entire disk → Done
8. Confirm → Continue
9. Profile setup:
   ```
   Your name: admin
   Server name: forexbot
   Username: admin
   Password: [secure password]
   ```
10. SSH Setup → **Install OpenSSH server** ✓
11. Featured snaps → Skip
12. Wait for installation...
13. **Reboot now**

### Step 4: Initial Ubuntu Setup

**After reboot, login via console or SSH:**

```bash
# SSH from your Windows machine
ssh admin@[vm-ip]

# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y curl git wget ca-certificates gnupg lsb-release

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
rm get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Logout and login again for docker group
exit
ssh admin@[vm-ip]

# Verify
docker --version
docker-compose --version
```

### Step 5: Deploy ForexSwing AI

```bash
# Create application directory
sudo mkdir -p /opt/forexswing-ai-2025
sudo chown $USER:$USER /opt/forexswing-ai-2025
cd /opt/forexswing-ai-2025
```

**Copy files from Windows:**

```powershell
# From Windows (PowerShell or Command Prompt)
scp -r "G:\Trading bot stuff\forexswing-ai-2025\*" admin@[vm-ip]:/opt/forexswing-ai-2025/
```

**Or use WinSCP, FileZilla, or create deployment package:**

```bash
# Back in VM
cd /opt/forexswing-ai-2025

# Make scripts executable
chmod +x deploy/setup_vm.sh quick_start.sh

# Setup environment
cp .env.example .env
nano .env
# Add your API keys

# Build and start
docker-compose build
docker-compose up -d

# Collect initial data
docker-compose run --rm data-collector python src/data/market_data_collector.py --once

# Check status
docker-compose ps
curl http://localhost:8082/api/status
```

### Step 6: Auto-Start on Boot

```bash
# Setup systemd service
sudo cp deploy/forexbot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable forexbot
sudo systemctl start forexbot

# Check status
sudo systemctl status forexbot
```

---

## VM Management from Proxmox

### Start/Stop/Restart

**From Proxmox Web UI:**
- Select VM → Right side: Start/Stop/Reboot buttons

**From Proxmox shell:**

```bash
qm start 100      # Start
qm stop 100       # Stop
qm shutdown 100   # Graceful shutdown
qm reboot 100     # Reboot
qm status 100     # Check status
```

### Backups

**Scheduled backups:**

```bash
# From Proxmox shell
vzdump 100 --compress zstd --mode snapshot --storage local

# Automated daily at 2 AM
cat > /etc/cron.d/forexbot-backup <<EOF
0 2 * * * root vzdump 100 --compress zstd --mode snapshot --storage local
EOF
```

### Snapshots

**From Proxmox Web UI:**
1. Select VM
2. Click "Snapshots" tab
3. Click "Take Snapshot"
4. Name it (e.g., "before-update")

**Restore snapshot:**
1. Select snapshot
2. Click "Rollback"

### Clone VM

1. Right-click VM → Clone
2. Mode: Full Clone
3. New VM ID: 101
4. Name: forexbot-test

---

## Performance Tuning

### Increase Resources

**From Proxmox Web UI:**
1. **Shutdown VM first**
2. Hardware tab:
   - Memory → Edit → 8192 MiB
   - Processors → Edit → 4 cores
   - Hard Disk → Resize → +10 GB

**From Proxmox shell:**

```bash
# Stop VM
qm stop 100

# Increase CPU
qm set 100 -cores 4

# Increase RAM
qm set 100 -memory 8192

# Increase disk (add 10GB)
qm resize 100 scsi0 +10G

# Start VM
qm start 100
```

**Inside VM after disk resize:**

```bash
# Expand filesystem
sudo growpart /dev/sda 3
sudo resize2fs /dev/sda3
df -h  # Verify
```

### Enable Qemu Guest Agent

**Inside VM:**

```bash
sudo apt install qemu-guest-agent
sudo systemctl enable qemu-guest-agent
sudo systemctl start qemu-guest-agent
```

**From Proxmox:**

```bash
qm set 100 -agent enabled=1
```

Provides better VM shutdown and more accurate resource reporting.

---

## Networking

### Static IP (Recommended)

**Inside VM:**

```bash
# Edit netplan config
sudo nano /etc/netplan/00-installer-config.yaml
```

```yaml
network:
  version: 2
  ethernets:
    ens18:
      addresses:
        - 192.168.1.100/24  # Your desired static IP
      gateway4: 192.168.1.1  # Your gateway
      nameservers:
        addresses:
          - 8.8.8.8
          - 8.8.4.4
```

```bash
# Apply
sudo netplan apply

# Verify
ip addr show ens18
```

### Port Forwarding (Access from WAN)

**From Proxmox host:**

```bash
# Forward port 8082 from Proxmox to VM
iptables -t nat -A PREROUTING -p tcp --dport 8082 -j DNAT --to-destination [vm-ip]:8082
iptables -t nat -A POSTROUTING -s [vm-ip] -j MASQUERADE

# Make persistent
apt install iptables-persistent
netfilter-persistent save
```

---

## Monitoring

### From Proxmox UI:
- Select VM → Summary
- View graphs: CPU, Memory, Network, Disk I/O

### Inside VM:

```bash
# Interactive menu
./quick_start.sh

# Real-time dashboard
python src/monitoring/dashboard.py

# Docker stats
docker stats

# System resources
htop
```

---

## Troubleshooting

### VM Won't Start

```bash
# Check status
qm status 100

# Check logs
journalctl -u qemu-server@100

# Force stop and restart
qm stop 100
qm start 100
```

### Network Issues

```bash
# Inside VM
ip addr
ping google.com

# Check Proxmox bridge
brctl show
```

### Disk Full

```bash
# Inside VM
df -h

# Clean Docker
docker system prune -a

# Clean apt cache
sudo apt clean
```

### Performance Issues

```bash
# Check CPU/RAM from Proxmox
qm status 100 --verbose

# Inside VM
htop
docker stats
```

---

## Migration

### To Another Proxmox Node

```bash
# Online migration (cluster only)
qm migrate 100 [target-node] --online

# Offline migration
qm stop 100
qm migrate 100 [target-node]
```

### Export/Import

```bash
# Export
vzdump 100 --dumpdir /var/lib/vz/dump

# Import on another Proxmox
qmrestore /var/lib/vz/dump/vzdump-qemu-100-*.vma.zst 100
```

---

## Security

### Firewall (Inside VM)

```bash
sudo apt install ufw
sudo ufw allow 22/tcp     # SSH
sudo ufw allow 8082/tcp   # API
sudo ufw enable
```

### Proxmox Firewall

**From Proxmox Web UI:**
1. Datacenter → Firewall → Enable
2. VM → Firewall → Add rules

---

## VM vs LXC Comparison

| Feature | VM | LXC Container |
|---------|-----|---------------|
| **RAM** | 4GB minimum | 2GB minimum |
| **Disk** | 20GB+ | 10GB+ |
| **Boot** | ~30 sec | <5 sec |
| **Isolation** | Full | Process-level |
| **Performance** | Good | Excellent |
| **Complexity** | Simple | Requires features |

**Recommendation: Use LXC unless you need full VM isolation**

---

## Quick Reference

```bash
# PROXMOX HOST
qm list                    # List VMs
qm start 100              # Start
qm stop 100               # Stop
qm status 100             # Status
vzdump 100                # Backup

# INSIDE VM
docker-compose up -d      # Start bot
docker-compose logs -f    # Logs
./quick_start.sh          # Menu
sudo systemctl status forexbot  # Service status
```

---

## Cost

**FREE** - Uses existing Proxmox server!

---

## Next Steps

1. ✅ Create VM in Proxmox
2. ✅ Install Ubuntu
3. ✅ Install Docker
4. ✅ Deploy ForexSwing AI
5. ✅ Configure API keys
6. ✅ Start services

**Your bot is now running 24/7 on Proxmox VM!**
