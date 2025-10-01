// ForexSwing AI Companion - Fixed Background Service Worker

console.log('🤖 ForexSwing AI Companion background script loading...');

// Extension installation and setup
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
    }
    
    // Create context menus
    try {
        chrome.contextMenus.create({
            id: 'analyzeCurrentPair',
            title: 'Analyze with ForexSwing AI',
            contexts: ['page'],
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
        
        console.log('✅ Context menus created');
    } catch (error) {
        console.error('❌ Context menu creation failed:', error);
    }
});

// Monitor tab updates for supported platforms
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    if (changeInfo.status === 'complete' && tab.url) {
        const supportedSites = [
            'tradingview.com',
            'forex.com',
            'oanda.com'
        ];
        
        const isSupported = supportedSites.some(site => tab.url.includes(site));
        
        if (isSupported) {
            console.log('✅ Supported trading platform loaded:', tab.url);
            
            // Update extension badge
            chrome.action.setBadgeText({
                tabId: tabId,
                text: 'AI'
            });
            
            chrome.action.setBadgeBackgroundColor({
                tabId: tabId,
                color: '#27ae60'
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

    switch (request.action) {
        case 'fetchAPI':
            // Proxy API calls to bypass CSP restrictions
            const apiUrl = request.url;
            console.log('🔄 Proxying API request:', apiUrl);

            fetch(apiUrl)
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`API error: ${response.status}`);
                    }
                    return response.json();
                })
                .then(data => {
                    console.log('✅ API response:', data);
                    sendResponse({ success: true, data: data });
                })
                .catch(error => {
                    console.error('❌ API fetch error:', error);
                    sendResponse({ success: false, error: error.message });
                });

            return true; // Keep channel open for async response

        case 'pairDetected':
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
            break;

        case 'analysisUpdate':
            console.log('✅ Analysis updated:', request.data);
            sendResponse({ success: true });
            break;

        case 'apiError':
            console.error('❌ API Error from content script:', request.error);

            chrome.action.setBadgeText({
                tabId: sender.tab.id,
                text: '⚠️'
            });

            chrome.action.setBadgeBackgroundColor({
                tabId: sender.tab.id,
                color: '#e74c3c'
            });

            sendResponse({ success: true });
            break;

        default:
            sendResponse({ success: false, error: 'Unknown action' });
    }

    return true; // Keep message channel open for async response
});

// Handle context menu clicks
chrome.contextMenus.onClicked.addListener((info, tab) => {
    console.log('Context menu clicked:', info.menuItemId);
    
    switch (info.menuItemId) {
        case 'analyzeCurrentPair':
            // Trigger analysis refresh in content script
            chrome.scripting.executeScript({
                target: { tabId: tab.id },
                function: () => {
                    if (window.forexAI && typeof window.forexAI.updateAnalysis === 'function') {
                        window.forexAI.updateAnalysis();
                        console.log('✅ Analysis refresh triggered');
                    } else {
                        console.warn('⚠️ ForexAI not found or updateAnalysis not available');
                    }
                }
            });
            break;
            
        case 'openDashboard':
            chrome.tabs.create({
                url: 'http://localhost:8082'
            });
            break;
    }
});

// Simplified API health check
async function checkApiHealth() {
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout
        
        const response = await fetch('http://localhost:8082/api/status', {
            method: 'GET',
            signal: controller.signal,
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        clearTimeout(timeoutId);
        
        const isHealthy = response.ok;
        
        // Store API health status
        chrome.storage.local.set({ 
            apiHealthy: isHealthy,
            lastHealthCheck: Date.now()
        });
        
        console.log('📡 API health check:', isHealthy ? '✅ OK' : '❌ Failed');
        
        return isHealthy;
        
    } catch (error) {
        console.error('❌ API health check failed:', error.message);
        chrome.storage.local.set({ 
            apiHealthy: false,
            lastHealthCheck: Date.now(),
            lastError: error.message
        });
        return false;
    }
}

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

// Check API health on startup and periodically
checkApiHealth();
setInterval(checkApiHealth, 5 * 60 * 1000); // Every 5 minutes

console.log('🚀 ForexSwing AI Companion background script ready');