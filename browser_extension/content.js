// ForexSwing AI Companion - Content Script
// Injects AI analysis overlay into trading platforms

class ForexAICompanion {
    constructor() {
        this.apiUrl = 'http://localhost:8082/api';
        this.overlay = null;
        this.currentPair = null;
        this.updateInterval = null;
        this.isInitialized = false;
        
        console.log('🤖 ForexSwing AI Companion loaded');
        this.initialize();
    }
    
    async initialize() {
        // Wait for page to load
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.setup());
        } else {
            this.setup();
        }
    }
    
    setup() {
        console.log('🚀 Setting up ForexSwing AI Companion');
        
        // Detect trading platform
        const platform = this.detectPlatform();
        console.log(`📊 Detected platform: ${platform}`);
        
        // Create overlay
        this.createOverlay();
        
        // Start monitoring for currency pairs
        this.startMonitoring();
        
        this.isInitialized = true;
    }
    
    detectPlatform() {
        const hostname = window.location.hostname;
        
        if (hostname.includes('tradingview.com')) {
            return 'tradingview';
        } else if (hostname.includes('forex.com')) {
            return 'forex.com';
        } else if (hostname.includes('oanda.com')) {
            return 'oanda';
        } else {
            return 'unknown';
        }
    }
    
    createOverlay() {
        // Create overlay container
        this.overlay = document.createElement('div');
        this.overlay.id = 'forexai-companion-overlay';
        this.overlay.className = 'forexai-overlay';
        this.overlay.innerHTML = `
            <div class="forexai-header">
                <span class="forexai-logo">🚀</span>
                <span class="forexai-title">Forex AI</span>
                <button class="forexai-refresh" id="forexai-refresh-btn" title="Analyze Current Pair">🔄</button>
                <button class="forexai-close" id="forexai-close-btn">×</button>
            </div>
            <div class="forexai-pair-selector">
                <select id="forexai-pair-select">
                    <option value="">Select Pair...</option>
                    <option value="EURUSD">EUR/USD</option>
                    <option value="GBPUSD">GBP/USD</option>
                    <option value="USDJPY">USD/JPY</option>
                    <option value="USDCHF">USD/CHF</option>
                    <option value="AUDUSD">AUD/USD</option>
                    <option value="USDCAD">USD/CAD</option>
                    <option value="NZDUSD">NZD/USD</option>
                </select>
            </div>
            <div class="forexai-content">
                <div class="forexai-loading">
                    <div class="forexai-spinner"></div>
                    <span>Select a pair or detecting...</span>
                </div>
            </div>
        `;
        
        // Add to page
        document.body.appendChild(this.overlay);
        
        // Add button event listeners
        this.overlay.querySelector('#forexai-close-btn').addEventListener('click', () => {
            this.overlay.style.display = 'none';
        });

        this.overlay.querySelector('#forexai-refresh-btn').addEventListener('click', () => {
            console.log('🔄 Manual refresh requested');
            if (this.currentPair) {
                this.updateAnalysis();
            } else {
                alert('Please select a currency pair first');
            }
        });

        // Add dropdown listener
        this.overlay.querySelector('#forexai-pair-select').addEventListener('change', (e) => {
            const selectedPair = e.target.value;
            if (selectedPair) {
                console.log(`📊 User selected pair: ${selectedPair}`);
                this.currentPair = selectedPair;

                const content = this.overlay.querySelector('.forexai-content');
                content.innerHTML = `
                    <div class="forexai-ready">
                        <div class="forexai-pair-detected">${selectedPair}</div>
                        <div class="forexai-instruction">Click 🔄 to analyze</div>
                    </div>
                `;
            }
        });
        
        // Make draggable
        this.makeDraggable();
        
        console.log('✅ AI overlay created');
    }
    
    makeDraggable() {
        const header = this.overlay.querySelector('.forexai-header');
        let isDragging = false;
        let currentX, currentY, initialX, initialY;
        
        header.addEventListener('mousedown', (e) => {
            isDragging = true;
            initialX = e.clientX - this.overlay.offsetLeft;
            initialY = e.clientY - this.overlay.offsetTop;
            document.addEventListener('mousemove', drag);
            document.addEventListener('mouseup', stopDrag);
        });
        
        const drag = (e) => {
            if (isDragging) {
                currentX = e.clientX - initialX;
                currentY = e.clientY - initialY;
                this.overlay.style.left = currentX + 'px';
                this.overlay.style.top = currentY + 'px';
            }
        };
        
        const stopDrag = () => {
            isDragging = false;
            document.removeEventListener('mousemove', drag);
            document.removeEventListener('mouseup', stopDrag);
        };
    }
    
    startMonitoring() {
        // Auto-detection disabled - user selects pair from dropdown
        console.log('✅ Ready - Select a currency pair from the dropdown');
    }
    
    // Detection methods removed - user selects pair manually from dropdown
    
    detectTradingViewPair() {
        // PRIORITY 1: Check URL first (fastest and most reliable)
        const urlParams = new URLSearchParams(window.location.search);
        const symbolParam = urlParams.get('symbol');
        if (symbolParam) {
            const cleanSymbol = symbolParam.split(':').pop();
            if (this.isValidForexPair(cleanSymbol)) {
                console.log(`✅ Detected from URL params: ${cleanSymbol}`);
                return cleanSymbol;
            }
        }

        // PRIORITY 2: Check URL path
        const urlMatch = window.location.pathname.match(/\/chart\/[^\/]*\/([^\/]+)/);
        if (urlMatch) {
            const symbol = urlMatch[1];
            if (this.isValidForexPair(symbol)) {
                console.log(`✅ Detected from URL path: ${symbol}`);
                return symbol;
            }
        }

        // PRIORITY 3: Quick DOM check (only fast selectors)
        const quickSelectors = [
            '[data-name="legend-source-title"]',
            '[data-symbol-short]',
            '.chart-container [class*="symbol"]',
            '[class*="symbolName"]'
        ];

        for (const selector of quickSelectors) {
            const element = document.querySelector(selector);
            if (element) {
                const text = element.textContent.trim();
                if (this.isValidForexPair(text)) {
                    console.log(`✅ Detected from DOM (${selector}): ${text}`);
                    return text;
                }
            }
        }

        // PRIORITY 4: Scan specific watchlist/ticker areas only (not entire page)
        const targetAreas = document.querySelectorAll('.symbolName-RsFlttSS, [class*="symbol"], .ticker');
        for (const area of targetAreas) {
            const text = area.textContent?.trim();
            if (text && text.length <= 20 && this.isValidForexPair(text)) {
                console.log(`✅ Detected from ticker area: ${text}`);
                return text;
            }
        }

        return null;
    }
    
    detectForexComPair() {
        // Look for currency pair in forex.com interface
        const selectors = [
            '.instrument-name',
            '.symbol-name',
            '.pair-name'
        ];
        
        for (const selector of selectors) {
            const element = document.querySelector(selector);
            if (element) {
                const text = element.textContent.trim();
                if (this.isValidForexPair(text)) {
                    return text;
                }
            }
        }
        
        return null;
    }
    
    detectOandaPair() {
        // Look for currency pair in OANDA interface
        const selectors = [
            '.instrument-title',
            '.symbol-display',
            '.currency-pair'
        ];
        
        for (const selector of selectors) {
            const element = document.querySelector(selector);
            if (element) {
                const text = element.textContent.trim();
                if (this.isValidForexPair(text)) {
                    return text;
                }
            }
        }
        
        return null;
    }
    
    isValidForexPair(text) {
        if (!text) return false;

        // Clean the text - remove everything except letters
        const cleaned = text.replace(/[^A-Z]/gi, '').toUpperCase();

        // Check if text matches forex pair pattern (6 letters, valid currencies)
        const forexPattern = /^(EUR|GBP|USD|JPY|CHF|AUD|CAD|NZD)(EUR|GBP|USD|JPY|CHF|AUD|CAD|NZD)$/;
        const isValid = forexPattern.test(cleaned) && cleaned.length === 6;

        // Only log when we find a valid pair
        if (isValid) {
            console.log(`✅ Valid forex pair found: "${text}" → "${cleaned}"`);
        }

        return isValid;
    }
    
    normalizePair(pair) {
        // Convert to standard format (EUR/USD)
        const cleaned = pair.replace(/[^A-Z]/gi, '').toUpperCase();
        if (cleaned.length === 6) {
            return `${cleaned.slice(0, 3)}/${cleaned.slice(3)}`;
        }
        return pair;
    }
    
    async updateAnalysis() {
        if (!this.currentPair) return;

        const content = this.overlay.querySelector('.forexai-content');

        // Step 1: Show initial status
        content.innerHTML = `
            <div class="forexai-loading">
                <div class="forexai-spinner"></div>
                <span>🔍 Fetching ${this.currentPair} data...</span>
            </div>
        `;

        try {
            const normalizedPair = this.normalizePair(this.currentPair);
            const apiUrl = `${this.apiUrl}/analyze?pair=${encodeURIComponent(normalizedPair)}`;

            // Step 2: Update status
            setTimeout(() => {
                content.innerHTML = `
                    <div class="forexai-loading">
                        <div class="forexai-spinner"></div>
                        <span>📊 Running AI analysis...</span>
                    </div>
                `;
            }, 500);

            // Use background script to fetch API (bypasses CSP)
            const response = await chrome.runtime.sendMessage({
                action: 'fetchAPI',
                url: apiUrl
            });

            if (!response.success) {
                throw new Error(response.error || 'API request failed');
            }

            // Step 3: Update status before displaying
            content.innerHTML = `
                <div class="forexai-loading">
                    <div class="forexai-spinner"></div>
                    <span>✅ Preparing results...</span>
                </div>
            `;

            setTimeout(() => {
                this.displayAnalysis(response.data);
            }, 200);

        } catch (error) {
            console.error('❌ Analysis failed:', error);
            this.displayError(error.message);
        }
    }
    
    displayAnalysis(analysis) {
        const content = this.overlay.querySelector('.forexai-content');
        
        // Determine action color
        const actionColor = analysis.action === 'BUY' ? '#27ae60' : 
                           analysis.action === 'SELL' ? '#e74c3c' : '#f39c12';
        
        // Determine risk color
        const riskColor = analysis.risk_level === 'LOW' ? '#27ae60' :
                         analysis.risk_level === 'HIGH' ? '#e74c3c' : '#f39c12';
        
        content.innerHTML = `
            <div class="forexai-analysis">
                <div class="forexai-pair">${analysis.pair}</div>
                
                <div class="forexai-recommendation">
                    <div class="forexai-action" style="color: ${actionColor}">
                        ${analysis.action} ${(analysis.confidence * 100).toFixed(0)}%
                    </div>
                    <div class="forexai-risk" style="color: ${riskColor}">
                        Risk: ${analysis.risk_level}
                    </div>
                </div>
                
                <div class="forexai-components">
                    <div class="forexai-component">
                        <span class="forexai-label">LSTM:</span>
                        <span class="forexai-value">${analysis.components.lstm}</span>
                    </div>
                    <div class="forexai-component">
                        <span class="forexai-label">Gemini:</span>
                        <span class="forexai-value">${analysis.components.gemini}</span>
                    </div>
                    <div class="forexai-component">
                        <span class="forexai-label">News:</span>
                        <span class="forexai-value">${analysis.components.news}</span>
                    </div>
                </div>
                
                <div class="forexai-meta">
                    <div class="forexai-agreements">
                        Agreements: ${analysis.agreements}/3
                    </div>
                    <div class="forexai-time">
                        ${analysis.processing_time}
                    </div>
                </div>
                
                <div class="forexai-actions">
                    <button class="forexai-refresh" id="forexai-refresh-btn">
                        🔄 Refresh
                    </button>
                    <button class="forexai-details" id="forexai-details-btn">
                        📊 Details
                    </button>
                </div>
            </div>
        `;
        
        // Add event listeners for action buttons
        const refreshBtn = this.overlay.querySelector('#forexai-refresh-btn');
        const detailsBtn = this.overlay.querySelector('#forexai-details-btn');
        
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.updateAnalysis());
        }
        if (detailsBtn) {
            detailsBtn.addEventListener('click', () => this.showDetails());
        }
    }
    
    displayError(message) {
        const content = this.overlay.querySelector('.forexai-content');
        content.innerHTML = `
            <div class="forexai-error">
                <div class="forexai-error-icon">⚠️</div>
                <div class="forexai-error-message">${message}</div>
                <button class="forexai-retry" id="forexai-retry-btn">
                    🔄 Retry
                </button>
            </div>
        `;
        
        // Add event listener for retry button
        const retryBtn = this.overlay.querySelector('#forexai-retry-btn');
        if (retryBtn) {
            retryBtn.addEventListener('click', () => this.updateAnalysis());
        }
    }
    
    async showDetails() {
        if (!this.currentPair) return;
        
        console.log('📊 Showing detailed analysis...');
        const content = this.overlay.querySelector('.forexai-content');
        
        // Show loading state
        content.innerHTML = `
            <div class="forexai-loading">
                <div class="forexai-spinner"></div>
                <span>Loading detailed analysis...</span>
            </div>
        `;
        
        try {
            // Get fresh analysis data
            const normalizedPair = this.normalizePair(this.currentPair);
            const response = await fetch(`${this.apiUrl}/analyze?pair=${encodeURIComponent(normalizedPair)}`);
            
            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }
            
            const analysis = await response.json();
            this.displayDetailedAnalysis(analysis);
            
        } catch (error) {
            console.error('❌ Failed to load details:', error);
            this.displayError('Failed to load detailed analysis');
        }
    }
    
    displayDetailedAnalysis(analysis) {
        const content = this.overlay.querySelector('.forexai-content');
        
        // Create detailed view
        content.innerHTML = `
            <div class="forexai-details">
                <div class="forexai-header-details">
                    <h3>${analysis.pair} - Detailed Analysis</h3>
                    <button class="forexai-back" id="forexai-back-btn">← Back</button>
                </div>
                
                <div class="forexai-main-signal">
                    <div class="forexai-action-large" style="color: ${analysis.action === 'BUY' ? '#27ae60' : analysis.action === 'SELL' ? '#e74c3c' : '#f39c12'}">
                        ${analysis.action}
                    </div>
                    <div class="forexai-confidence-large">${(analysis.confidence * 100).toFixed(1)}% Confidence</div>
                    <div class="forexai-risk-large">Risk: ${analysis.risk_level}</div>
                </div>
                
                <div class="forexai-components-detailed">
                    <h4>AI Component Analysis:</h4>
                    <div class="forexai-component-item">
                        <span class="forexai-component-name">🧠 LSTM Neural Network:</span>
                        <span class="forexai-component-value">${analysis.components.lstm}</span>
                    </div>
                    <div class="forexai-component-item">
                        <span class="forexai-component-name">🤖 Google Gemini AI:</span>
                        <span class="forexai-component-value">${analysis.components.gemini}</span>
                    </div>
                    <div class="forexai-component-item">
                        <span class="forexai-component-name">📰 News Sentiment:</span>
                        <span class="forexai-component-value">${analysis.components.news}</span>
                    </div>
                </div>
                
                <div class="forexai-meta-detailed">
                    <div class="forexai-agreements-detailed">
                        <strong>AI Consensus:</strong> ${analysis.agreements}/3 models agree
                    </div>
                    <div class="forexai-processing-detailed">
                        <strong>Processing Time:</strong> ${analysis.processing_time}
                    </div>
                    <div class="forexai-timestamp-detailed">
                        <strong>Generated:</strong> ${new Date(analysis.timestamp).toLocaleTimeString()}
                    </div>
                    <div class="forexai-score-detailed">
                        <strong>Raw Score:</strong> ${analysis.score.toFixed(4)}
                    </div>
                </div>
                
                <div class="forexai-data-sources">
                    <h4>Data Sources:</h4>
                    <div class="forexai-sources-list">
                        ${analysis.data_sources.map(source => `<span class="forexai-source-tag">${source}</span>`).join('')}
                    </div>
                </div>
            </div>
        `;
        
        // Add back button event listener
        const backBtn = this.overlay.querySelector('#forexai-back-btn');
        if (backBtn) {
            backBtn.addEventListener('click', () => this.displayAnalysis(analysis));
        }
    }
    
    destroy() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        
        if (this.overlay) {
            this.overlay.remove();
        }
        
        console.log('🛑 ForexSwing AI Companion destroyed');
    }
}

// Initialize companion when script loads
let forexAI = null;

// Wait for page to be ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeCompanion);
} else {
    initializeCompanion();
}

function initializeCompanion() {
    // Only initialize once
    if (!forexAI) {
        forexAI = new ForexAICompanion();
        
        // Make globally accessible for debugging
        window.forexAI = forexAI;
    }
}

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    if (forexAI) {
        forexAI.destroy();
    }
});

console.log('📄 ForexSwing AI Companion content script loaded');