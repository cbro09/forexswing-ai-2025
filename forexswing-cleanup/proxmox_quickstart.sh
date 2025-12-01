#!/bin/bash
# ForexSwing AI 2025 - Proxmox Quick Start
# Run inside your LXC container

set -e

echo "========================================"
echo "  ForexSwing AI 2025 - Quick Setup"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Please run with sudo${NC}"
    exit 1
fi

# Step 1: Update system
echo -e "\n${YELLOW}[1/5] Updating system...${NC}"
apt-get update -qq
apt-get upgrade -y -qq

# Step 2: Install Docker
echo -e "\n${YELLOW}[2/5] Installing Docker...${NC}"
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
    echo -e "${GREEN}Docker installed${NC}"
else
    echo -e "${GREEN}Docker already installed${NC}"
fi

# Step 3: Install Docker Compose
echo -e "\n${YELLOW}[3/5] Installing Docker Compose...${NC}"
if ! command -v docker-compose &> /dev/null; then
    apt-get install -y -qq docker-compose-plugin
    ln -sf /usr/libexec/docker/cli-plugins/docker-compose /usr/local/bin/docker-compose 2>/dev/null || true
    echo -e "${GREEN}Docker Compose installed${NC}"
else
    echo -e "${GREEN}Docker Compose already installed${NC}"
fi

# Step 4: Setup directories
echo -e "\n${YELLOW}[4/5] Setting up directories...${NC}"
mkdir -p data/MarketData data/models data/News data/logs logs

# Step 5: Environment check
echo -e "\n${YELLOW}[5/5] Checking environment...${NC}"
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo -e "${YELLOW}Created .env from example - EDIT WITH YOUR API KEYS${NC}"
    else
        echo -e "${RED}No .env.example found!${NC}"
    fi
else
    echo -e "${GREEN}.env exists${NC}"
fi

# Final status
echo ""
echo "========================================"
echo -e "${GREEN}  Setup Complete!${NC}"
echo "========================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Edit your API keys:"
echo "   nano .env"
echo ""
echo "2. Build and start:"
echo "   docker-compose up -d --build"
echo ""
echo "3. Check status:"
echo "   docker-compose ps"
echo "   curl http://localhost:8082/api/status"
echo ""
echo "4. View logs:"
echo "   docker-compose logs -f"
echo ""
