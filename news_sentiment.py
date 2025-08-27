import requests
import pandas as pd
from transformers import pipeline
import streamlit as st
import json # Import json for pretty printing
import time # For exponential backoff

FINNHUB_API_KEY = "Your API key here"

# Cache the sentiment model to avoid reloading it on every rerun
@st.cache_resource
def load_sentiment_model():
    """
    Loads a pre-trained sentiment analysis model.
    """
    try:
        # Using a smaller, faster model for sentiment analysis
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    except Exception as e:
        st.error(f"Error loading sentiment model: {e}. Please check your internet connection or try again.")
        return None

# Load the sentiment model once
sentiment_analyzer = load_sentiment_model()

def fetch_news(ticker, api_key, retries=3, backoff_factor=0.5):
    """
    Fetches news articles for a given ticker using Finnhub.io API.
    Performs sentiment analysis on the recent 10 articles.
    Implements exponential backoff for API calls.
    
    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL').
        api_key (str): Your Finnhub.io API key.
        retries (int): Number of times to retry the API call.
        backoff_factor (float): Factor by which to increase delay between retries.
        
    Returns:
        tuple: A tuple containing:
               - list: A list of dictionaries, each representing a news article with
                       title, content, link, sentiment, and sentiment score.
               - dict: An overall sentiment summary (positive_count, negative_count, neutral_count, total_articles).
    """
    if not api_key:
        st.warning("Finnhub API key is missing. Please ensure it's correctly set in news_sentiment.py.")
        return [], {"total_articles": 0, "positive_count": 0, "negative_count": 0, "neutral_count": 0}

    # Finnhub News API endpoint for company news
    # Fetch news from 30 days ago to today
    from_date = (pd.to_datetime('today') - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
    to_date = pd.to_datetime('today').strftime('%Y-%m-%d')

    url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={from_date}&to={to_date}&token={api_key}"
    
    print(f"\n--- Debugging Finnhub News Fetch for {ticker} ---")
    print(f"Request URL: {url}")

    news_data = []
    for i in range(retries):
        try:
            response = requests.get(url)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            news_data = response.json()
            
            print(f"Finnhub API Response Status Code: {response.status_code}")
            print(f"Raw Finnhub API Response (first 500 chars): {json.dumps(news_data, indent=2)[:500]}...")
            break # Break loop if successful
        except requests.exceptions.RequestException as e:
            wait_time = backoff_factor * (2 ** i)
            print(f"Error fetching news from Finnhub (attempt {i+1}/{retries}): {e}. Retrying in {wait_time:.2f} seconds.")
            time.sleep(wait_time)
        except ValueError as e:
            print(f"Error decoding JSON from Finnhub news response (ValueError): {e}")
            st.error("Error decoding JSON from Finnhub news response. API key might be invalid or response format unexpected.")
            return [], {"total_articles": 0, "positive_count": 0, "negative_count": 0, "neutral_count": 0}
    
    if not news_data:
        st.error(f"Failed to fetch news from Finnhub after {retries} attempts. Please check your API key and internet connection.")
        return [], {"total_articles": 0, "positive_count": 0, "negative_count": 0, "neutral_count": 0}

    # --- IMPORTANT CHANGE: Process only the recent 10 articles for sentiment calculation ---
    # Finnhub usually returns news ordered by published date (most recent first)
    recent_10_articles = news_data[:10] 
    print(f"Processing sentiment for {len(recent_10_articles)} recent articles.")

    articles_with_sentiment = []
    positive_count = 0
    negative_count = 0
    neutral_count = 0

    if recent_10_articles: # Iterate over the sliced list
        for article in recent_10_articles:
            title = article.get('headline', 'No Title') # Finnhub uses 'headline'
            content = article.get('summary', title)     # Finnhub uses 'summary'
            url = article.get('url', '#')               # Finnhub uses 'url'

            sentiment_label = "neutral"
            sentiment_score = 0.5

            if sentiment_analyzer and content:
                try:
                    # Analyze sentiment of the news article content
                    # Limit input length for the sentiment model
                    sentiment_result = sentiment_analyzer(content[:512])[0]
                    sentiment_label = sentiment_result['label'].lower() # Ensure lowercase
                    sentiment_score = sentiment_result['score']
                except Exception as e:
                    print(f"Warning: Could not analyze sentiment for an article: {e}")
                    # Keep default neutral sentiment if analysis fails
            
            if sentiment_label == "positive":
                positive_count += 1
            elif sentiment_label == "negative":
                negative_count += 1
            else:
                neutral_count += 1

            articles_with_sentiment.append({
                'title': title,
                'content': content,
                'link': url,
                'sentiment': sentiment_label,
                'score': sentiment_score
            })
    else:
        print(f"No news data received from Finnhub for ticker {ticker}.")
        st.info(f"No news articles found for {ticker} using Finnhub API. Check if the ticker is valid or if your API key has access.")

    # Overall summary now reflects only the recent_10_articles
    overall_summary = {
        "total_articles": len(articles_with_sentiment),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count
    }

    print(f"--- End Debugging Finnhub News Fetch for {ticker} ---")
    return articles_with_sentiment, overall_summary

def summarize_article_gemini(text_to_summarize):
    """
    Summarizes text using the Gemini API (gemini-2.5-flash-preview-05-20).
    """
    try:
        # Gemini API configuration
        api_key = "Your API key here" 
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"

        prompt = f"Summarize the following news article concisely, focusing on the main points:\n\n{text_to_summarize}"
        
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}]
        }

        headers = {'Content-Type': 'application/json'}

        # Implement exponential backoff for Gemini API calls
        max_retries = 3
        for i in range(max_retries):
            try:
                response = requests.post(api_url, headers=headers, data=json.dumps(payload))
                response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
                break # Break loop if successful
            except requests.exceptions.RequestException as e:
                wait_time = 2 ** i # Exponential backoff
                if i < max_retries - 1:
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Failed to summarize text after {max_retries} retries due to network or API issue: {e}")
        
        gemini_result = response.json()
        
        if gemini_result.get('candidates') and len(gemini_result['candidates']) > 0 and \
           gemini_result['candidates'][0].get('content') and \
           gemini_result['candidates'][0]['content'].get('parts') and \
           len(gemini_result['candidates'][0]['content']['parts']) > 0:
            
            summary = gemini_result['candidates'][0]['content']['parts'][0]['text']
            return summary
        else:
            raise Exception("Gemini API did not return a valid summary.")

    except Exception as e:
        raise Exception(f"An unexpected error occurred during summarization: {e}")

