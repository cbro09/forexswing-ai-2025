# ForexSwing AI 2025 - Cleanup Guide

## 🎯 Goal: Lean, Mean, Trading Machine

This guide removes bloat while keeping everything functional. Follow in order.

---

## 🚨 CRITICAL FIX FIRST: API Key Exposure

Your `.env.example` contains a **real** Alpha Vantage key. Fix immediately:

```bash
# In .env.example, change:
ALPHA_VANTAGE_KEY=OXGW647WZO8XTKA1

# To:
ALPHA_VANTAGE_KEY=your_alpha_vantage_key_here
```

**Then revoke and regenerate your key at:** https://www.alphavantage.co/support/#api-key

---

## 🗑️ FILES TO DELETE

### Redundant API Service
```bash
rm companion_api_service_lite.py
```
**Why:** You have two API services doing the same thing. Keep the full one.

### Duplicate/Redundant Deployment Docs
```bash
rm deploy/VM_DEPLOYMENT_SUMMARY.md
rm deploy/QUICK_REFERENCE.md
```
**Why:** Info is duplicated across 4 guides. Keep only:
- `DEPLOYMENT.md` (main guide)
- `PROXMOX_LXC_DEPLOYMENT.md` (your target)
- `NATIVE_UBUNTU_DEPLOYMENT.md` (backup option)

### App Launcher (Unnecessary Wrapper)
```bash
rm app_launcher.py
```
**Why:** Just run `streamlit run forexswing_app.py` directly. This wrapper adds nothing.

### Old/Test Files in Root (if present)
```bash
rm -f quick_test.py
rm -f debug_signals.py
rm -f demo_enhanced_gemini.py
rm -f test_with_real_data.py
rm -f extended_paper_trading_test.py
rm -f comprehensive_system_test.py
```
**Why:** Tests belong in `/tests/` folder, not root. Move if needed, delete if redundant.

---

## 📁 RECOMMENDED STRUCTURE (After Cleanup)

```
forexswing-ai-2025/
├── ForexBot.py                    # Core AI engine ✅ KEEP
├── easy_forex_bot.py              # User interface ✅ KEEP
├── paper_trading_system.py        # Testing system ✅ KEEP
├── enhanced_gemini_trading_system.py  # Multi-AI ✅ KEEP
├── companion_api_service.py       # API (single version) ✅ KEEP
├── forexswing_app.py              # Streamlit dashboard (optional)
│
├── Models/
│   └── ForexLSTM.py               # LSTM architecture ✅ KEEP
│
├── src/
│   ├── core/models/               # ✅ KEEP
│   ├── integrations/              # ✅ KEEP
│   ├── indicators/                # ✅ KEEP
│   └── manual_trading_system.py   # ✅ KEEP (useful for MT5)
│
├── data/
│   ├── MarketData/                # ✅ KEEP (CSV price data)
│   ├── models/                    # ✅ KEEP (trained weights)
│   └── News/                      # ✅ KEEP (sentiment data)
│
├── deploy/
│   ├── DEPLOYMENT.md              # Main guide ✅ KEEP
│   ├── PROXMOX_LXC_DEPLOYMENT.md  # Your target ✅ KEEP
│   ├── NATIVE_UBUNTU_DEPLOYMENT.md # Backup ✅ KEEP
│   ├── setup_vm.sh                # ✅ KEEP
│   └── forexbot.service           # ✅ KEEP
│
├── browser_extension/             # 🟡 OPTIONAL (delete if not using TradingView)
│
├── tests/                         # ✅ KEEP (but don't deploy to server)
│
├── scripts/
│   ├── system_check.py            # ✅ KEEP
│   └── install_dependencies.py    # ✅ KEEP
│
├── .claude/
│   └── CLAUDE.md                  # ✅ KEEP (great for continuity)
│
├── docker-compose.yml             # ✅ KEEP
├── Dockerfile                     # ✅ KEEP
├── requirements.txt               # ✅ KEEP
├── .env.example                   # ✅ KEEP (after fixing API key!)
├── .gitignore                     # ✅ KEEP
└── README.md                      # ✅ KEEP
```

---

## 🧹 CLEANUP COMMANDS (Copy & Run)

```bash
# Navigate to your repo
cd /path/to/forexswing-ai-2025

# 1. Remove redundant files
rm -f companion_api_service_lite.py
rm -f app_launcher.py
rm -f deploy/VM_DEPLOYMENT_SUMMARY.md
rm -f deploy/QUICK_REFERENCE.md

# 2. Move test files to tests/ (if in root)
mkdir -p tests/misc
mv -f quick_test.py tests/misc/ 2>/dev/null
mv -f debug_signals.py tests/misc/ 2>/dev/null
mv -f demo_enhanced_gemini.py tests/misc/ 2>/dev/null
mv -f test_with_real_data.py tests/misc/ 2>/dev/null
mv -f extended_paper_trading_test.py tests/misc/ 2>/dev/null
mv -f comprehensive_system_test.py tests/misc/ 2>/dev/null

# 3. Optional: Remove browser extension if not needed
# rm -rf browser_extension/

# 4. Clean Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete

# 5. Verify cleanup
echo "Remaining files:"
find . -maxdepth 1 -type f -name "*.py" | wc -l
```

---

## 📊 BEFORE vs AFTER

| Metric | Before | After |
|--------|--------|-------|
| Root .py files | ~15+ | 6 |
| Deployment docs | 5 | 3 |
| API services | 2 | 1 |
| Test files in root | ~6 | 0 |

---

## 🚀 NEXT STEPS AFTER CLEANUP

1. **Fix the API key** (critical security issue)
2. **Run the cleanup commands** above
3. **Test core functionality:**
   ```bash
   python scripts/system_check.py
   python ForexBot.py
   python easy_forex_bot.py
   ```
4. **Deploy to Proxmox** using the lean setup

---

## 💡 OPTIONAL: Browser Extension Decision

**Keep if:** You actively use TradingView and want the overlay
**Delete if:** You'll just use the API/Streamlit dashboard

```bash
# To remove browser extension
rm -rf browser_extension/
```

This removes ~10 files and simplifies the project.

---

## ✅ Validation Checklist

After cleanup, verify these work:

- [ ] `python ForexBot.py` - Core AI runs
- [ ] `python easy_forex_bot.py` - Interface works
- [ ] `python paper_trading_system.py` - Paper trading functional
- [ ] `python companion_api_service.py` - API starts on :8082
- [ ] `docker-compose build` - Docker builds successfully
- [ ] `python scripts/system_check.py` - System validation passes

---

**Questions?** The core functionality is untouched - we're just removing duplicates and organizing.
