# ForexSwing AI - Arch Linux Installation Guide

## 🚀 **Quick Installation Commands**

Run these commands to install all dependencies on Arch Linux:

### **Core Python Packages**
```bash
# Install core dependencies via pacman (recommended)
sudo pacman -S python-requests python-feedparser python-pandas python-numpy python-pytorch

# Install additional packages
sudo pacman -S python-pip python-setuptools
```

### **Google Gemini API (via pip)**
```bash
# Install Gemini API package (not available in Arch repos)
pip install --user google-generativeai
```

### **Alternative: Using AUR Helper (yay)**
```bash
# If you use yay AUR helper
yay -S python-google-generativeai
```

## 📦 **Package Mapping**

| **Required Package** | **Arch Package Name** | **Purpose** |
|---------------------|----------------------|-------------|
| requests | `python-requests` | HTTP API calls |
| feedparser | `python-feedparser` | Yahoo Finance RSS |
| pandas | `python-pandas` | Data manipulation |
| numpy | `python-numpy` | Numerical computing |
| torch | `python-pytorch` | LSTM model |
| google-generativeai | `pip install google-generativeai` | Gemini AI API |

## ⚡ **One-Line Installation**
```bash
sudo pacman -S python-requests python-feedparser python-pandas python-numpy python-pytorch && pip install --user google-generativeai
```

## 🧪 **Test Installation**
```bash
python -c "
import requests, feedparser, pandas, numpy, torch
print('✅ Core packages installed!')
try:
    import google.generativeai
    print('✅ Gemini API available!')
except ImportError:
    print('⚠️ Run: pip install --user google-generativeai')
"
```

## 🔧 **Manual Installation (if needed)**
```bash
# Create user-local installation directory
mkdir -p ~/.local/lib/python3.13/site-packages

# Install via pip with --user flag
pip install --user requests feedparser pandas numpy torch google-generativeai
```

## 🚀 **Start ForexSwing AI**
```bash
# After installation, start the companion system:
python companion_api_service.py

# Should show:
# 🤖 FOREXSWING AI - COMPANION API SERVICE
# 🚀 Companion API server starting on http://localhost:8080
```

## 🌐 **Browser Extension**
1. Open Chrome → Extensions → Developer mode
2. Click "Load unpacked" 
3. Select `browser_extension` folder
4. Visit TradingView → See AI overlay!

## 🔑 **API Keys Already Configured**
- ✅ **Alpha Vantage**: `OXGW647WZO8XTKA1`
- ✅ **Google Gemini**: `AIzaSyBcSKWOjFghU3UItDVjNL62tWpGnn7I-bQ`

## 🆘 **If You Get Errors**

### **"externally-managed-environment"**
```bash
# Use --user flag for pip installs
pip install --user google-generativeai
```

### **Permission denied**
```bash
# Use sudo for pacman
sudo pacman -S python-pandas python-numpy

# Use --user for pip
pip install --user google-generativeai
```

### **Package not found**
```bash
# Update package database
sudo pacman -Sy

# Try alternative names
sudo pacman -S python-requests python-feedparser
```

## ✅ **Success Verification**

After installation, you should be able to run:
```bash
curl "http://localhost:8080/api/status"
# Returns: {"status": "operational", "supported_pairs": 7}

curl "http://localhost:8080/api/analyze?pair=EUR/USD"  
# Returns: Real-time AI analysis with Multi-AI fusion
```

## 🎯 **Why Arch Packages Are Better**

- ✅ **System integration** - Properly managed by pacman
- ✅ **No virtual environments** - Works with system Python
- ✅ **Automatic updates** - Updated with system
- ✅ **Dependencies resolved** - Pacman handles conflicts
- ✅ **Performance optimized** - Compiled for your system

**Your ForexSwing AI companion will be fully operational after running these commands!** 🚀