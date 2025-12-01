#!/bin/bash
# ForexSwing AI 2025 - Automated Cleanup Script
# Run from the root of your forexswing-ai-2025 directory

set -e

echo "========================================"
echo "  ForexSwing AI 2025 - Cleanup Tool"
echo "========================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Safety check
if [ ! -f "ForexBot.py" ]; then
    echo -e "${RED}Error: Run this from the forexswing-ai-2025 root directory${NC}"
    exit 1
fi

# Confirm
echo ""
echo -e "${YELLOW}This will remove redundant files and organize the project.${NC}"
echo "Core functionality will NOT be affected."
echo ""
read -p "Continue? (y/N) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Count files before
BEFORE=$(find . -maxdepth 1 -type f -name "*.py" | wc -l)

echo ""
echo -e "${YELLOW}[1/6] Removing redundant API service...${NC}"
rm -f companion_api_service_lite.py && echo "  Removed: companion_api_service_lite.py" || true

echo -e "${YELLOW}[2/6] Removing unnecessary launcher...${NC}"
rm -f app_launcher.py && echo "  Removed: app_launcher.py" || true

echo -e "${YELLOW}[3/6] Cleaning up duplicate deployment docs...${NC}"
rm -f deploy/VM_DEPLOYMENT_SUMMARY.md && echo "  Removed: deploy/VM_DEPLOYMENT_SUMMARY.md" || true
rm -f deploy/QUICK_REFERENCE.md && echo "  Removed: deploy/QUICK_REFERENCE.md" || true

echo -e "${YELLOW}[4/6] Moving test files to tests/...${NC}"
mkdir -p tests/misc
for file in quick_test.py debug_signals.py demo_enhanced_gemini.py test_with_real_data.py extended_paper_trading_test.py comprehensive_system_test.py; do
    if [ -f "$file" ]; then
        mv "$file" tests/misc/ && echo "  Moved: $file -> tests/misc/"
    fi
done

echo -e "${YELLOW}[5/6] Cleaning Python cache...${NC}"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
echo "  Cleaned __pycache__ and .pyc files"

echo -e "${YELLOW}[6/6] Checking .env.example for exposed keys...${NC}"
if grep -q "OXGW647WZO8XTKA1" .env.example 2>/dev/null; then
    echo -e "${RED}  WARNING: Real API key found in .env.example!${NC}"
    echo -e "${RED}  Please replace with 'your_alpha_vantage_key_here'${NC}"
    echo -e "${RED}  Then regenerate your key at Alpha Vantage!${NC}"
else
    echo -e "${GREEN}  .env.example looks clean${NC}"
fi

# Count files after
AFTER=$(find . -maxdepth 1 -type f -name "*.py" | wc -l)

echo ""
echo "========================================"
echo -e "${GREEN}  Cleanup Complete!${NC}"
echo "========================================"
echo ""
echo "Results:"
echo "  Root .py files: $BEFORE -> $AFTER"
echo ""
echo "Remaining essential files:"
ls -1 *.py 2>/dev/null | head -10
echo ""
echo -e "${GREEN}Run validation:${NC}"
echo "  python scripts/system_check.py"
echo "  python ForexBot.py"
echo ""
