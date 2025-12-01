#!/bin/bash
# Quick Start Script for ForexSwing AI 2025
# Simplifies common operations

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

show_menu() {
    clear
    echo "=========================================="
    echo "  ForexSwing AI 2025 - Quick Start Menu"
    echo "=========================================="
    echo ""
    echo "1)  Start all services"
    echo "2)  Stop all services"
    echo "3)  Restart all services"
    echo "4)  View logs (all services)"
    echo "5)  View logs (bot only)"
    echo "6)  Check system status"
    echo ""
    echo "7)  Collect market data (one-time)"
    echo "8)  Train/Retrain model"
    echo "9)  Monitor dashboard"
    echo ""
    echo "10) Setup environment (.env)"
    echo "11) Rebuild Docker images"
    echo "12) Clean up (remove containers)"
    echo ""
    echo "13) Test API connection"
    echo "14) Backup data"
    echo ""
    echo "0)  Exit"
    echo ""
    echo -n "Select option: "
}

start_services() {
    echo -e "${GREEN}Starting ForexSwing AI services...${NC}"
    docker-compose up -d
    echo -e "${GREEN}✅ Services started${NC}"
    echo ""
    echo "Access API at: http://localhost:8082"
    echo "Check status: curl http://localhost:8082/api/status"
}

stop_services() {
    echo -e "${YELLOW}Stopping ForexSwing AI services...${NC}"
    docker-compose down
    echo -e "${GREEN}✅ Services stopped${NC}"
}

restart_services() {
    echo -e "${YELLOW}Restarting ForexSwing AI services...${NC}"
    docker-compose restart
    echo -e "${GREEN}✅ Services restarted${NC}"
}

view_logs_all() {
    echo -e "${GREEN}Viewing logs (Ctrl+C to exit)...${NC}"
    docker-compose logs -f
}

view_logs_bot() {
    echo -e "${GREEN}Viewing bot logs (Ctrl+C to exit)...${NC}"
    docker-compose logs -f forexbot
}

check_status() {
    echo -e "${GREEN}Checking system status...${NC}"
    echo ""
    echo "Docker Containers:"
    docker-compose ps
    echo ""
    echo "API Status:"
    curl -s http://localhost:8082/api/status | python3 -m json.tool || echo "❌ API not responding"
}

collect_data() {
    echo -e "${GREEN}Collecting market data (this may take 15-20 minutes)...${NC}"
    docker-compose run --rm data-collector python src/data/market_data_collector.py --once
    echo -e "${GREEN}✅ Data collection complete${NC}"
}

train_model() {
    echo -e "${GREEN}Training model (this may take 30-60 minutes)...${NC}"
    docker-compose --profile training run --rm model-trainer
    echo -e "${GREEN}✅ Model training complete${NC}"
}

monitor_dashboard() {
    echo -e "${GREEN}Starting monitoring dashboard...${NC}"
    python3 src/monitoring/dashboard.py
}

setup_env() {
    if [ -f .env ]; then
        echo -e "${YELLOW}.env file already exists${NC}"
        echo -n "Overwrite? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            return
        fi
    fi

    cp .env.example .env
    echo -e "${GREEN}✅ Created .env file${NC}"
    echo ""
    echo "Please edit .env with your API keys:"
    echo "  nano .env"
    echo ""
    echo "Required keys:"
    echo "  - ALPHA_VANTAGE_KEY (https://www.alphavantage.co/support/#api-key)"
    echo "  - GOOGLE_API_KEY (https://makersuite.google.com/app/apikey)"
}

rebuild_images() {
    echo -e "${YELLOW}Rebuilding Docker images...${NC}"
    docker-compose build --no-cache
    echo -e "${GREEN}✅ Images rebuilt${NC}"
}

cleanup() {
    echo -e "${RED}This will remove all containers and volumes${NC}"
    echo -n "Are you sure? (y/N): "
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        docker-compose down -v
        echo -e "${GREEN}✅ Cleanup complete${NC}"
    else
        echo "Cancelled"
    fi
}

test_api() {
    echo -e "${GREEN}Testing API connection...${NC}"
    echo ""

    echo "1. Status endpoint:"
    curl -s http://localhost:8082/api/status | python3 -m json.tool || echo "❌ Failed"
    echo ""

    echo "2. Pairs endpoint:"
    curl -s http://localhost:8082/api/pairs | python3 -m json.tool || echo "❌ Failed"
    echo ""

    echo "3. Analysis endpoint (EUR/USD):"
    curl -s "http://localhost:8082/api/analyze?pair=EUR/USD" | python3 -m json.tool || echo "❌ Failed"
}

backup_data() {
    timestamp=$(date +%Y%m%d_%H%M%S)
    backup_file="forexbot_backup_${timestamp}.tar.gz"

    echo -e "${GREEN}Creating backup...${NC}"
    tar -czf "$backup_file" data/ models/ config/ .env 2>/dev/null || true
    echo -e "${GREEN}✅ Backup created: ${backup_file}${NC}"
    echo "Size: $(du -h "$backup_file" | cut -f1)"
}

# Main loop
while true; do
    show_menu
    read -r choice

    case $choice in
        1) start_services ;;
        2) stop_services ;;
        3) restart_services ;;
        4) view_logs_all ;;
        5) view_logs_bot ;;
        6) check_status ;;
        7) collect_data ;;
        8) train_model ;;
        9) monitor_dashboard ;;
        10) setup_env ;;
        11) rebuild_images ;;
        12) cleanup ;;
        13) test_api ;;
        14) backup_data ;;
        0) echo "Goodbye!"; exit 0 ;;
        *) echo -e "${RED}Invalid option${NC}" ;;
    esac

    echo ""
    echo -n "Press Enter to continue..."
    read -r
done
