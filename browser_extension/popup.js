// ForexSwing AI Companion - Popup Script

const API_URL = 'http://localhost:8082/api';

// Initialize popup
document.addEventListener('DOMContentLoaded', async () => {
    console.log('🤖 ForexSwing AI Companion popup loaded');
    await checkApiStatus();
    await loadSupportedPairs();
    
    // Add event listeners for buttons
    document.getElementById('toggleOverlayBtn').addEventListener('click', toggleOverlay);
    document.getElementById('refreshAnalysisBtn').addEventListener('click', refreshAnalysis);
    document.getElementById('openDashboardBtn').addEventListener('click', openDashboard);
    document.getElementById('openSettingsBtn').addEventListener('click', openSettings);
    
    // Add event listeners for footer links
    document.getElementById('helpLink').addEventListener('click', openHelp);
    document.getElementById('aboutLink').addEventListener('click', openAbout);
    document.getElementById('reportIssueLink').addEventListener('click', reportIssue);
});

// Check API service status
async function checkApiStatus() {
    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');
    
    try {
        const response = await fetch(`${API_URL}/status`, { 
            method: 'GET',
            headers: { 'Content-Type': 'application/json' }
        });
        
        if (response.ok) {
            const status = await response.json();
            statusDot.style.background = '#27ae60';
            statusText.textContent = `Connected - ${status.supported_pairs} pairs available`;
        } else {
            throw new Error(`HTTP ${response.status}`);
        }
        
    } catch (error) {
        console.error('API status check failed:', error);
        statusDot.style.background = '#e74c3c';
        statusText.textContent = 'API service offline';
    }
}

// Load supported currency pairs
async function loadSupportedPairs() {
    const pairsList = document.getElementById('pairsList');
    
    try {
        const response = await fetch(`${API_URL}/pairs`);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const data = await response.json();
        const pairs = data.supported_pairs;
        
        pairsList.innerHTML = '';
        
        pairs.forEach(pair => {
            const pairItem = document.createElement('div');
            pairItem.className = 'pair-item';
            pairItem.innerHTML = `
                <span class="pair-name">${pair}</span>
                <span class="pair-status">Ready</span>
            `;
            pairsList.appendChild(pairItem);
        });
        
    } catch (error) {
        console.error('Failed to load pairs:', error);
        pairsList.innerHTML = `
            <div style="text-align: center; color: #e74c3c; font-size: 11px;">
                Failed to load pairs
            </div>
        `;
    }
}

// Toggle overlay visibility
async function toggleOverlay() {
    try {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        
        await chrome.scripting.executeScript({
            target: { tabId: tab.id },
            function: () => {
                const overlay = document.getElementById('forexai-companion-overlay');
                if (overlay) {
                    overlay.style.display = overlay.style.display === 'none' ? 'block' : 'none';
                } else {
                    console.log('Overlay not found - may need to reload page');
                }
            }
        });
        
    } catch (error) {
        console.error('Failed to toggle overlay:', error);
    }
}

// Refresh analysis
async function refreshAnalysis() {
    try {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        
        await chrome.scripting.executeScript({
            target: { tabId: tab.id },
            function: () => {
                if (window.forexAI) {
                    window.forexAI.updateAnalysis();
                } else {
                    console.log('ForexAI companion not found');
                }
            }
        });
        
        // Visual feedback
        const btn = event.target.closest('.action-btn');
        const originalText = btn.innerHTML;
        btn.innerHTML = '<span class="action-icon">✅</span>Refreshed';
        
        setTimeout(() => {
            btn.innerHTML = originalText;
        }, 1000);
        
    } catch (error) {
        console.error('Failed to refresh analysis:', error);
    }
}

// Open dashboard
function openDashboard() {
    chrome.tabs.create({ url: 'http://localhost:8082' });
}

// Open settings
function openSettings() {
    // For now, just show an alert - could expand to options page
    alert('Settings functionality coming soon!\\n\\nCurrent API: ' + API_URL);
}

// Open help
function openHelp() {
    const helpText = `
🤖 ForexSwing AI Companion Help

GETTING STARTED:
1. Make sure the API service is running (green dot)
2. Visit a supported trading platform (TradingView, Forex.com, OANDA)
3. The AI overlay will automatically detect currency pairs
4. View real-time AI analysis and recommendations

SUPPORTED PLATFORMS:
• TradingView (tradingview.com)
• Forex.com
• OANDA

FEATURES:
• Real-time AI analysis (LSTM + Gemini + News)
• Multi-AI consensus building
• Risk level assessment
• Live news sentiment integration

TROUBLESHOOTING:
• Red status dot: Start the API service
• No overlay: Reload the trading platform page
• No analysis: Check if currency pair is supported

API SERVICE:
Run: python companion_api_service.py
URL: http://localhost:8082
    `;
    
    alert(helpText);
}

// Open about
function openAbout() {
    const aboutText = `
🤖 ForexSwing AI Companion v1.0.0

PROFESSIONAL-GRADE AI TRADING COMPANION

✅ 55.2% LSTM Model Accuracy
✅ Multi-AI Decision Fusion
✅ Live News Sentiment Analysis
✅ Real-time Risk Assessment

TECHNOLOGIES:
• Deep Learning (PyTorch LSTM)
• Google Gemini AI Integration
• Multi-source News Analysis
• Advanced Technical Indicators

DEVELOPED BY: ForexSwing AI Team
COPYRIGHT: 2025 ForexSwing AI
LICENSE: Proprietary

For more information visit our documentation.
    `;
    
    alert(aboutText);
}

// Report issue
function reportIssue() {
    const issueTemplate = `
FOREXSWING AI COMPANION - ISSUE REPORT

Please describe your issue:
- What were you trying to do?
- What happened instead?
- What browser and platform are you using?
- Is the API service running?

System Information:
- Extension Version: 1.0.0
- Browser: ${navigator.userAgent}
- Platform: ${navigator.platform}
- API Status: ${document.getElementById('statusText').textContent}

Please send this information to: support@forexswing-ai.com
    `;
    
    // Copy to clipboard
    navigator.clipboard.writeText(issueTemplate).then(() => {
        alert('Issue report template copied to clipboard!\\n\\nPlease email it to: support@forexswing-ai.com');
    }).catch(() => {
        alert(issueTemplate);
    });
}

// Auto-refresh status every 30 seconds
setInterval(checkApiStatus, 30000);