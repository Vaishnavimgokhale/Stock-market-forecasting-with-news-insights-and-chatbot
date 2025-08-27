import requests
import json
import streamlit as st
import pandas as pd
from datetime import timedelta, datetime
import time # For exponential backoff

# The Canvas environment will inject the actual API key at runtime.
GEMINI_API_KEY = "Your gemini API key here"

def get_gemini_response(prompt_text, chat_history):
    """
    Sends a prompt to the Gemini API and returns the response.
    Includes chat history for context.
    Implements exponential backoff for API calls.
    
    Args:
        prompt_text (str): The current user's query.
        chat_history (list): List of dictionaries containing previous chat turns.
                              Each dict should have "role" ("user" or "model") and "content".
                              
    Returns:
        str: The generated response from Gemini, or an error message.
    """
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={GEMINI_API_KEY}"

    gemini_chat_history = []
    for chat_entry in chat_history:
        gemini_chat_history.append({"role": chat_entry["role"], "parts": [{"text": chat_entry["content"]}]})

    # Add the current user prompt
    gemini_chat_history.append({"role": "user", "parts": [{"text": prompt_text}]})

    payload = {
        "contents": gemini_chat_history
    }

    headers = {'Content-Type': 'application/json'}

    max_retries = 3
    for i in range(max_retries):
        try:
            response = requests.post(api_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            gemini_result = response.json()
            
            if gemini_result.get('candidates') and len(gemini_result['candidates']) > 0 and \
               gemini_result['candidates'][0].get('content') and \
               gemini_result['candidates'][0]['content'].get('parts') and \
               len(gemini_result['candidates'][0]['content']['parts']) > 0:
                
                return gemini_result['candidates'][0]['content']['parts'][0]['text']
            else:
                # Log the full API response for debugging if no valid candidate is found
                print(f"Gemini API response without valid candidate: {json.dumps(gemini_result, indent=2)}")
                return "I couldn't get a valid response from Gemini. The model might not have generated a suitable answer."

        except requests.exceptions.RequestException as e:
            wait_time = 2 ** i # Exponential backoff
            print(f"Error calling Gemini API (attempt {i+1}/{max_retries}): {e}. Retrying in {wait_time} seconds.")
            time.sleep(wait_time)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from Gemini API response: {e}. Response text: {response.text}")
            return "Error parsing Gemini's response. Please try again."
        except Exception as e:
            print(f"An unexpected error occurred during Gemini API call: {e}")
            return f"An unexpected error occurred: {e}"
    
    return "Failed to get a response from Gemini after multiple retries. Please check your internet connection or try again later."


def generate_chatbot_response(user_query, current_ticker, current_df, ensemble_df, news_articles_data, overall_sentiment_summary, chat_history):
    """
    Generates a response from the Gemini-backed chatbot based on user query and app state.
    
    Args:
        user_query (str): The question asked by the user.
        current_ticker (str): The currently selected stock ticker.
        current_df (pd.DataFrame): Historical stock data.
        ensemble_df (pd.DataFrame): Forecasted stock data.
        news_articles_data (list): List of news articles with sentiment.
        overall_sentiment_summary (dict): Summary of news sentiment.
        chat_history (list): Current chat history for conversational context.

    Returns:
        str: The generated chatbot response.
    """
    context_info = f"You are a stock market analysis chatbot. The current stock ticker being analyzed is {current_ticker}.\n\n"

    if current_df is not None and not current_df.empty:
        last_close_price = current_df['Close'].iloc[-1]
        context_info += f"The last known closing price for {current_ticker} is {last_close_price:.2f} on {current_df['Date'].iloc[-1].strftime('%Y-%m-%d')}.\n"
        context_info += f"Historical data available from {current_df['Date'].min().strftime('%Y-%m-%d')} to {current_df['Date'].max().strftime('%Y-%m-%d')}.\n"
    
    if ensemble_df is not None and not ensemble_df.empty:
        forecast_period = st.session_state.current_forecast_days # Access from session state as it's a global app parameter
        forecast_start = ensemble_df['Date'].min().strftime('%Y-%m-%d')
        forecast_end = ensemble_df['Date'].max().strftime('%Y-%m-%d')
        latest_forecast_price = ensemble_df['Ensemble Forecast'].iloc[-1]
        context_info += f"A forecast for the next {forecast_period} days is available, predicting prices up to {forecast_end}. The latest ensemble forecast price is {latest_forecast_price:.2f}.\n"

    if overall_sentiment_summary:
        context_info += f"Recent news sentiment for {current_ticker} shows: {overall_sentiment_summary['positive_count']} positive, {overall_sentiment_summary['negative_count']} negative, and {overall_sentiment_summary['neutral_count']} neutral articles out of {overall_sentiment_summary['total_articles']} recent articles.\n"
        if news_articles_data:
            context_info += "Titles of recent articles: " + ", ".join([a['title'] for a in news_articles_data[:5]]) + ".\n"
    
    context_info += "\nBased on the above context, answer the user's question. If the question is outside the scope of stock analysis or current data, politely decline."
    
    full_prompt = f"{context_info}\n\nUser: {user_query}"
    
    response = get_gemini_response(full_prompt, chat_history)
    return response
