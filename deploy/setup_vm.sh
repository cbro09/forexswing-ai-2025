#!/bin/bash
# ForexSwing AI 2025 - VM Setup Script
# Run this script on a fresh Ubuntu/Debian VM to set up the trading bot

set -e

echo "=========================================="
echo "ForexSwing AI 2025 - VM Setup"
echo "=========================================="

# Update system
echo "Updating system..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker
echo "Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
    echo "Docker installed successfully"
else
    echo "Docker already installed"
fi

# Install Docker Compose
echo "Installing Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo "Docker Compose installed successfully"
else
    echo "Docker Compose already installed"
fi

# Install additional dependencies
echo "Installing additional dependencies..."
sudo apt-get install -y \
    git \
    python3 \
    python3-pip \
    curl \
    vim \
    htop \
    tmux

# Create application directory
echo "Creating application directory..."
sudo mkdir -p /opt/forexswing-ai-2025
sudo chown -R $USER:$USER /opt/forexswing-ai-2025

# Clone or copy repository
echo "Setting up application files..."
if [ -d ".git" ]; then
    echo "Copying files from current directory..."
    cp -r . /opt/forexswing-ai-2025/
else
    echo "Please manually copy your application files to /opt/forexswing-ai-2025/"
fi

cd /opt/forexswing-ai-2025

# Create necessary directories
mkdir -p data/models data/MarketData data/News data/logs config

# Setup environment file
if [ ! -f .env ]; then
    echo "Creating .env file from example..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your API keys!"
    echo "   Run: nano /opt/forexswing-ai-2025/.env"
fi

# Build Docker images
echo "Building Docker images..."
docker-compose build

# Setup systemd service
echo "Setting up systemd service..."
sudo cp deploy/forexbot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable forexbot

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit your .env file with API keys:"
echo "   nano /opt/forexswing-ai-2025/.env"
echo ""
echo "2. Start the services:"
echo "   sudo systemctl start forexbot"
echo ""
echo "3. Check status:"
echo "   sudo systemctl status forexbot"
echo "   docker-compose logs -f"
echo ""
echo "4. Access the API:"
echo "   http://your-vm-ip:8082"
echo ""
echo "5. Collect initial data:"
echo "   docker-compose run data-collector python src/data/market_data_collector.py --once"
echo ""
echo "6. (Optional) Train model:"
echo "   docker-compose --profile training run model-trainer"
echo ""
echo "=========================================="
