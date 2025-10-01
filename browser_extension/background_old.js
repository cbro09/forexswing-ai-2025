// ForexSwing AI Companion - Background Service Worker

console.log('🤖 ForexSwing AI Companion background script loaded');

// Extension installation
chrome.runtime.onInstalled.addListener((details) => {
    console.log('ForexSwing AI Companion installed:', details.reason);
    
    if (details.reason === 'install') {
        // Show welcome notification
        chrome.notifications.create({
            type: 'basic',
            iconUrl: 'icons/icon48.png',
            title: 'ForexSwing AI Companion',
            message: 'Extension installed! Visit TradingView to see AI analysis.'
        });
        
        // Open welcome tab
        chrome.tabs.create({
            url: 'http://localhost:8082'
        });
    }
});

// Extension icon click is handled by popup.html in manifest
// No need for onClicked listener when using default_popup

// Monitor tab updates
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    if (changeInfo.status === 'complete' && tab.url) {
        // Check if it's a supported trading platform
        const supportedSites = [
            'tradingview.com',
            'forex.com',
            'oanda.com'
        ];
        
        const isSupported = supportedSites.some(site => tab.url.includes(site));
        
        if (isSupported) {
            console.log('Supported trading platform loaded:', tab.url);
            
            // Update extension badge
            chrome.action.setBadgeText({
                tabId: tabId,
                text: 'AI'
            });
            
            chrome.action.setBadgeBackgroundColor({
                tabId: tabId,
                color: '#27ae60'
            });
            
            // Optional: Show notification for first visit
            chrome.storage.local.get(['firstVisit'], (result) => {
                if (!result.firstVisit) {
                    chrome.notifications.create({
                        type: 'basic',
                        iconUrl: 'icons/icon48.png',
                        title: 'ForexSwing AI Active',
                        message: 'AI analysis overlay is now active on this page!'
                    });
                    
                    chrome.storage.local.set({ firstVisit: true });
                }
            });
        } else {
            // Clear badge for non-supported sites
            chrome.action.setBadgeText({
                tabId: tabId,
                text: ''
            });
        }
    }
});

// Handle messages from content scripts
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    console.log('Background received message:', request);
    
    if (request.action === 'pairDetected') {
        // Update badge with detected pair
        chrome.action.setBadgeText({
            tabId: sender.tab.id,
            text: request.pair.split('/')[0] // Show first currency
        });
        
        chrome.action.setBadgeBackgroundColor({
            tabId: sender.tab.id,
            color: '#667eea'
        });
        
        sendResponse({ success: true });
    }
    
    if (request.action === 'analysisUpdate') {
        // Could store analysis data or send notifications
        console.log('Analysis updated:', request.data);
        sendResponse({ success: true });
    }
    
    if (request.action === 'apiError') {
        // Handle API errors
        console.error('API Error:', request.error);
        
        chrome.action.setBadgeText({
            tabId: sender.tab.id,
            text: '⚠️'
        });
        
        chrome.action.setBadgeBackgroundColor({
            tabId: sender.tab.id,
            color: '#e74c3c'
        });
        
        sendResponse({ success: true });
    }
});

// Periodic API health check
async function checkApiHealth() {
    try {
        const response = await fetch('http://localhost:8082/api/status');
        const isHealthy = response.ok;
        
        // Store API health status
        chrome.storage.local.set({ apiHealthy: isHealthy });
        
        console.log('API health check:', isHealthy ? 'OK' : 'Failed');
        
    } catch (error) {
        console.error('API health check failed:', error);
        chrome.storage.local.set({ apiHealthy: false });
    }
}

// Check API health every 5 minutes
setInterval(checkApiHealth, 5 * 60 * 1000);

// Initial health check
checkApiHealth();

// Handle notification clicks
chrome.notifications.onClicked.addListener((notificationId) => {
    console.log('Notification clicked:', notificationId);
    
    // Close the notification
    chrome.notifications.clear(notificationId);
    
    // Open dashboard
    chrome.tabs.create({
        url: 'http://localhost:8082'
    });
});

// Context menu (right-click) actions
chrome.runtime.onInstalled.addListener(() => {
    // Add context menu for manual analysis trigger
    chrome.contextMenus.create({
        id: 'analyzeCurrentPair',
        title: 'Analyze with ForexSwing AI',
        contexts: ['selection', 'page'],
        documentUrlPatterns: [
            'https://www.tradingview.com/*',
            'https://tradingview.com/*',
            'https://www.forex.com/*',
            'https://www.oanda.com/*'
        ]
    });
    
    chrome.contextMenus.create({
        id: 'openDashboard',
        title: 'Open AI Dashboard',
        contexts: ['action']
    });
});

// Handle context menu clicks
chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === 'analyzeCurrentPair') {
        // Trigger analysis refresh
        chrome.scripting.executeScript({
            target: { tabId: tab.id },
            function: () => {
                if (window.forexAI) {
                    window.forexAI.updateAnalysis();
                }
            }
        });
    }
    
    if (info.menuItemId === 'openDashboard') {
        chrome.tabs.create({
            url: 'http://localhost:8082'
        });
    }
});

console.log('🚀 ForexSwing AI Companion background script ready');