# API Keys Configuration

## Alpha Vantage News & Market Data API
- **API Key**: `OXGW647WZO8XTKA1`
- **Free Tier**: 500 requests/day
- **Features**: Live news sentiment, forex data, technical indicators
- **Documentation**: https://www.alphavantage.co/documentation/

## Google Gemini AI API
- **API Key**: `AIzaSyBcSKWOjFghU3UItDVjNL62tWpGnn7I-bQ`
- **Free Tier**: 15 requests/minute, 1500 requests/day
- **Features**: Advanced market sentiment analysis, trading recommendations
- **Documentation**: https://ai.google.dev/gemini-api/docs

## Setup Instructions

### For News Integration:
```python
# In news_sentiment_analyzer.py
analyzer = MultiSourceNewsAnalyzer(
    alpha_vantage_key="OXGW647WZO8XTKA1",
    news_api_key=None  # Optional - NewsAPI key if available
)
```

### For Enhanced Gemini Integration:
```python
# In enhanced_news_gemini.py
interpreter = EnhancedNewsGeminiInterpreter(
    alpha_vantage_key="OXGW647WZO8XTKA1",
    gemini_api_key="AIzaSyBcSKWOjFghU3UItDVjNL62tWpGnn7I-bQ"
)
```

### For Direct Gemini API (Recommended):
```python
import google.generativeai as genai
genai.configure(api_key="AIzaSyBcSKWOjFghU3UItDVjNL62tWpGnn7I-bQ")
```

## Required Dependencies
```bash
pip install requests feedparser pandas numpy torch google-generativeai
```

## Usage Limits
- **Alpha Vantage**: 500 calls/day (free tier)
- **Google Gemini**: 15 requests/minute, 1500 requests/day (free tier)
- **Yahoo Finance RSS**: Unlimited (no auth required)
- **NewsAPI**: 1000 calls/day (if key provided)

## Security Note
Keep API keys secure and never commit them to public repositories.