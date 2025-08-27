import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta, datetime
import logging
from pytickersymbols import PyTickerSymbols
import json # Required for JSON handling in API calls
import requests 

# Import functions from custom modules
from model_trainer import train_and_forecast_models, evaluate_models
from data_handler import load_data
from news_sentiment import fetch_news, load_sentiment_model, FINNHUB_API_KEY, summarize_article_gemini
from chatbot import generate_chatbot_response # Import the chatbot function

# Suppress Prophet's verbose output
logging.getLogger('prophet').setLevel(logging.WARNING)

# --- Configuration ---
STOCKS_LIST_CACHE_TTL = 3600 * 24 # Cache stock list for 24 hours
POPULAR_TICKERS = ['AAPL', 'MSFT', 'GOOG', 'NVDA', 'AMZN', 'TSLA', 'META', 'NFLX'] # Curated list for initial display
MARKET_WATCH_ITEMS = [
    {'name': 'NIFTY 50', 'ticker': '^NSEI'}, # Nifty 50 Index
    {'name': 'Bitcoin', 'ticker': 'BTC-USD'},
    {'name': 'INR/USD', 'ticker': 'INR=X'},
    {'name': 'Ethereum', 'ticker': 'ETH-USD'},
    {'name': 'BNB', 'ticker': 'BNB-USD'},
    {'name': 'S&P 500', 'ticker': '^GSPC'},
    {'name': 'NASDAQ Comp.', 'ticker': '^IXIC'},
    {'name': 'Dow Jones', 'ticker': '^DJI'},
    {'name': 'Cardano', 'ticker': 'ADA-USD'},
]

# Minimum data points required for meaningful model training (e.g., 2 months of daily data)
MIN_DATA_FOR_MODELS = 60 

# --- Inject Font Awesome (separated for better rendering) ---
st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    """,
    unsafe_allow_html=True
)

# --- Inject Custom CSS ---
st.markdown(
    f'''
    <style>
    .stApp {{
        /* No background-image property here to use Streamlit's default background */
        background-size: cover;
        background-position: center center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    /* Overall Sidebar Styling: Slightly different background than main content */
    [data-testid="stSidebar"] {{
        background-color: #f0f2f6; /* A very light grey for sidebar background */
        color: var(--text-color); /* Use Streamlit variable for text color */
    }}

    /* Remove the default padding/margin of the very first div inside the sidebar */
    /* This targets the blank space at the very top */
    [data-testid="stSidebar"] > div:first-child {{
        padding-top: 0 !important;
        margin-top: 0 !important;
    }}
    
    /* Custom styles for the specific containers holding input parameters and navigation */
    .sidebar-input-box, .sidebar-nav-box {{
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 20px;
        background-color: white; /* Explicitly white background for these boxes */
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }}

    /* Ensure Streamlit's internal input elements within our custom boxes also have the correct background */
    /* Target specific Streamlit elements by their data-testid or class within our custom boxes */
    .sidebar-input-box [data-testid="stDateInput"] .stDateInput,
    .sidebar-input-box [data-testid="stSelectbox"] .stSelectbox,
    .sidebar-input-box [data-testid="stSlider"] .stSlider,
    .sidebar-input-box [data-testid="stTextInput"] > div > div > input, /* Target the actual input element */
    .sidebar-nav-box [data-testid="stSelectbox"] .stSelectbox {{
        background-color: white !important; /* Force white background for widgets inside boxes */
        border-radius: 5px;
    }}

    /* Specific rule to hide the hidden text input used for JS communication (if any custom JS is used elsewhere) */
    [data-testid="stSidebar"] [data-testid="stTextInput"] input[id*="hidden_nav_input_value"] {{
        display: none !important;
        height: 0 !important;
        visibility: hidden !important;
        padding: 0 !important;
        margin: 0 !important;
    }}

    /* Styling for individual news article containers */
    .news-article-container {{
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        background-color: var(--secondary-background-color); /* Use secondary background for article boxes */
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }}
    .news-article-container h5 {{
        margin-top: 0;
        margin-bottom: 8px;
        color: var(--text-color);
    }}
    .news-article-container p {{
        font-size: 0.9em;
        margin-bottom: 5px;
        color: var(--text-color);
    }}
    .news-article-container a {{
        color: #6a5acd; /* Link color */
        text-decoration: none;
    }}
    .news-article-container a:hover {{
        text-decoration: underline;
    }}
    .news-article-container .sentiment-positive {{ color: green; font-weight: bold; }}
    .news-article-container .sentiment-negative {{ color: red; font-weight: bold; }}
    .news-article-container .sentiment-neutral {{ color: orange; font-weight: bold; }}


    /* Additional style to ensure News Sentiment button in main content is visible */
    .summarize-button {{
        background-color: #6a5acd;
        color: white;
        padding: 8px 12px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-size: 0.9em;
        margin-top: 10px;
        transition: background-color 0.2s;
        opacity: 1; /* Ensure this button is not transparent */
    }}
    .summarize-button:hover {{
        background-color: #5548a8;
    }}
    .summarize-button:disabled {{
        background-color: #cccccc;
        cursor: not-allowed;
    }}
    .summary-output {{
        background-color: var(--secondary-background-color); /* Use Streamlit variable */
        border-left: 4px solid #6a5acd; /* Corrected border-left width */
        padding: 10px 15px;
        margin-top: 10px;
        border-radius: 4px;
        font-size: 0.85em;
        color: var(--text-color); /* Use Streamlit variable */
    }}
    .overall-summary-box {{
        background-color: var(--secondary-background-color); /* Use Streamlit variable */
        border: 1px solid var(--border-color); /* Use Streamlit variable */
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }}
    .overall-summary-box h4 {{
        color: var(--text-color); /* Use Streamlit variable */
        margin-top: 0;
        margin-bottom: 10px;
    }}
    .overall-summary-box p {{
        margin-bottom: 5px;
        font-size: 0.95em;
    }}
    .overall-summary-box .positive-count {{ color: green; font-weight: bold; }}
    .overall-summary-box .negative-count {{ color: red; font-weight: bold; }}
    .overall-summary-box .neutral-count {{ color: orange; font-weight: bold; }}
    </style>
    ''',
    unsafe_allow_html=True
)

@st.cache_data(ttl=3600 * 24)
def get_all_stock_tickers():
    """
    Fetches a comprehensive list of stock tickers using pytickersymbols.
    Caches the result to avoid repeated heavy calls.
    """
    stock_data = PyTickerSymbols()
    
    all_tickers = set()
    
    # Iterate through all available stocks and extract Yahoo Finance tickers
    # This is a more robust way to get a wide range of tickers across pytickersymbols versions
    for stock in stock_data.get_all_stocks():
        if 'yahoo_ticker' in stock and stock['yahoo_ticker']:
            all_tickers.add(stock['yahoo_ticker'])
        # Also add tickers from specific exchanges if available (redundant if get_all_stocks is comprehensive, but safe)
        if 'indices' in stock:
            for index_info in stock['indices']:
                if 'yahoo_ticker' in index_info and index_info['yahoo_ticker']:
                    all_tickers.add(index_info['yahoo_ticker'])
        if 'eod_ticker' in stock and stock['eod_ticker']: # EODHD tickers might also be useful
            all_tickers.add(stock['eod_ticker'])


    # Add popular tickers and market watch items to the full list
    all_tickers.update(POPULAR_TICKERS) 
    all_tickers.update([item['ticker'] for item in MARKET_WATCH_ITEMS]) # Add tickers from MARKET_WATCH_ITEMS

    # Sort for consistent display
    sorted_tickers = sorted(list(all_tickers))
    return sorted_tickers

# --- Streamlit UI Setup ---
st.set_page_config(layout="wide", page_title="Stock Forecast and News Insights")

st.title("📈 Stock Forecast and News Insights")
st.markdown("""
    This application forecasts stock prices using an ensembled method of Prophet and XGBoost models,
    and provides sentiment analysis for related news articles.
""")

# --- Initialize session state variables ---
if 'primary_ticker_selected_from_dropdown' not in st.session_state:
    st.session_state.primary_ticker_selected_from_dropdown = None # Stores the final selected primary ticker

if 'selected_page_id' not in st.session_state:
    st.session_state.selected_page_id = "Market Trends" # Default selection

if 'analysis_triggered_params' not in st.session_state:
    # Stores a tuple (ticker, start_date, end_date, forecast_days) for last successful analysis
    st.session_state.analysis_triggered_params = None

# Variables to store results of analysis
if 'current_df' not in st.session_state:
    st.session_state.current_df = None
if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = None
if 'current_forecast_days' not in st.session_state:
    st.session_state.current_forecast_days = None
if 'prophet_forecast' not in st.session_state:
    st.session_state.prophet_forecast = None
if 'xgb_future_predictions' not in st.session_state:
    st.session_state.xgb_future_predictions = None
if 'ensemble_df' not in st.session_state:
    st.session_state.ensemble_df = None
if 'predictions_xgb' not in st.session_state:
    st.session_state.predictions_xgb = None
if 'y_test_xgb' not in st.session_state:
    st.session_state.y_test_xgb = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'train_size_xgb' not in st.session_state:
    st.session_state.train_size_xgb = None
if 'LOOK_BACK' not in st.session_state:
    st.session_state.LOOK_BACK = None

# News sentiment specific session states
if 'news_articles_data' not in st.session_state:
    st.session_state.news_articles_data = None
if 'last_news_ticker_fetched' not in st.session_state:
    st.session_state.last_news_ticker_fetched = None
if 'summaries_cache' not in st.session_state:
    st.session_state.summaries_cache = {} # Cache for Gemini summaries
if 'overall_sentiment_summary' not in st.session_state:
    st.session_state.overall_sentiment_summary = None # Cache for overall sentiment

# Chatbot session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# --- Centralized Function to Run Analysis ---
def run_full_analysis(ticker_symbol, start_date_str, end_date_str, forecast_days_val):
    """
    Loads data, trains models, and stores results in session state.
    """
    # Only run if ticker is valid
    if not ticker_symbol:
        st.error("Please select a valid ticker to run the analysis.")
        return

    current_params = (ticker_symbol, start_date_str, end_date_str, forecast_days_val)

    # Check if analysis has already been performed for these parameters
    if st.session_state.analysis_triggered_params == current_params:
        return # No need to re-run if already done for these parameters

    status_message_area = st.empty()
    
    with status_message_area:
        st.info(f"Loading historical data for {ticker_symbol}...")
        df = load_data(ticker_symbol, start_date_str, end_date_str)
    
    status_message_area.empty()

    if df.empty:
        st.error("Failed to load data for the selected ticker. Please check the ticker symbol and date range.")
        # Clear analysis-related session states to indicate failure
        st.session_state.current_df = None
        st.session_state.analysis_triggered_params = None
        return
    
    # --- NEW: Check for minimum data length before model training ---
    if len(df) < MIN_DATA_FOR_MODELS:
        st.warning(f"Not enough historical data ({len(df)} days) for {ticker_symbol} to train models effectively. "
                   f"Please select a wider date range (at least {MIN_DATA_FOR_MODELS} days recommended). "
                   "Forecasting and metrics might be skipped or inaccurate.")
        # Clear model-related session states to reflect insufficient data
        st.session_state.prophet_forecast = None
        st.session_state.xgb_future_predictions = None
        st.session_state.ensemble_df = None
        st.session_state.predictions_xgb = None
        st.session_state.y_test_xgb = None
        st.session_state.scaler = None
        st.session_state.train_size_xgb = None
        st.session_state.LOOK_BACK = None
        st.session_state.current_df = df # Keep df for historical tab
        st.session_state.current_ticker = ticker_symbol
        st.session_state.current_forecast_days = forecast_days_val
        st.session_state.analysis_triggered_params = current_params # Mark as attempted
        return # Exit function, models won't be trained

    with status_message_area:
        st.info("Training models and generating predictions...")
        try:
            prophet_forecast, xgb_future_predictions, ensemble_df, predictions_xgb, y_test_xgb, scaler, train_size_xgb, LOOK_BACK_val = \
                train_and_forecast_models(df.copy(), forecast_days_val)
        except Exception as e:
            st.error(f"Error during model training: {e}. Please try adjusting inputs.")
            st.session_state.current_df = None
            st.session_state.analysis_triggered_params = None
            return
            
    status_message_area.empty()
    st.success("Analysis complete! Data and forecasts are ready.")

    # Store all results in session state
    st.session_state.current_df = df
    st.session_state.current_ticker = ticker_symbol
    st.session_state.current_forecast_days = forecast_days_val
    st.session_state.prophet_forecast = prophet_forecast
    st.session_state.xgb_future_predictions = xgb_future_predictions
    st.session_state.ensemble_df = ensemble_df
    st.session_state.predictions_xgb = predictions_xgb
    st.session_state.y_test_xgb = y_test_xgb
    st.session_state.scaler = scaler
    st.session_state.train_size_xgb = train_size_xgb
    st.session_state.LOOK_BACK = LOOK_BACK_val # Store the LOOK_BACK value used
    
    # Mark analysis as successfully triggered for these parameters
    st.session_state.analysis_triggered_params = current_params

    # Reset news data and summaries cache when new analysis is run
    st.session_state.news_articles_data = None
    st.session_state.last_news_ticker_fetched = None
    st.session_state.summaries_cache = {}
    st.session_state.overall_sentiment_summary = None

    # Clear chat history when new analysis is run
    st.session_state.chat_history = []


# --- Sidebar for User Inputs ---
# Using st.sidebar.header and st.sidebar.subheader directly inside the markdown div
st.sidebar.header("Input Parameters")

# Get the list of available tickers (cached)
available_tickers = get_all_stock_tickers()

# --- Single Stock Ticker Selection ---
st.sidebar.subheader("Stock Ticker") 

# Text input for typing the search query
search_query = st.sidebar.text_input(
    "Enter ticker or search for suggestions:",
    placeholder="e.g., AAPL, MSFT, GOOG",
    key="ticker_search_query"
).upper()

# Add a small, explicit label to guide the user to the suggestions dropdown
st.sidebar.markdown("---") # Separator for clarity
st.sidebar.markdown("**Select from suggestions:**")

# Determine options for the selectbox based on search_query
selectbox_options = []
if search_query:
    filtered_by_query = [
        ticker for ticker in available_tickers 
        if search_query in ticker
    ]
    # Prioritize exact match at the top if it exists
    if search_query in filtered_by_query:
        filtered_by_query.remove(search_query)
        filtered_by_query.insert(0, search_query)
    
    MAX_SUGGESTIONS = 50
    if len(filtered_by_query) > MAX_SUGGESTIONS:
        selectbox_options = filtered_by_query[:MAX_SUGGESTIONS]
        selectbox_options.append(f"... {len(filtered_by_query) - MAX_SUGGESTIONS} more matches")
    else:
        selectbox_options = filtered_by_query
    
    if not selectbox_options:
        selectbox_options = ["No matches found. Try a different ticker."]
else:
    selectbox_options = POPULAR_TICKERS

# Determine default index for selectbox
default_index = 0
if st.session_state.primary_ticker_selected_from_dropdown and st.session_state.primary_ticker_selected_from_dropdown in selectbox_options:
    default_index = selectbox_options.index(st.session_state.primary_ticker_selected_from_dropdown)
elif search_query and search_query in selectbox_options:
    default_index = selectbox_options.index(search_query)
elif not search_query and POPULAR_TICKERS and POPULAR_TICKERS[0] in selectbox_options: # Default to first popular if no search
    default_index = selectbox_options.index(POPULAR_TICKERS[0])
elif not selectbox_options:
    default_index = None

# The selectbox for suggestions
selected_ticker_from_dropdown = st.sidebar.selectbox(
    "Select ticker from suggestions:", # This label is now visible
    options=selectbox_options,
    index=default_index if default_index is not None else 0,
    key="ticker_final_selection",
    label_visibility="collapsed", # Set to collapsed to hide the default label
    on_change=lambda: st.session_state.update(
        primary_ticker_selected_from_dropdown=st.session_state.ticker_final_selection,
        analysis_triggered_params=None, # Invalidate previous analysis if ticker changes
        news_articles_data=None, # Clear news data on ticker change
        last_news_ticker_fetched=None,
        summaries_cache={} # Clear summaries cache
    )
)

# Final determination of the ticker symbol to use for the app logic
ticker_symbol_for_analysis = None
if selected_ticker_from_dropdown and not selected_ticker_from_dropdown.startswith("..."):
    if selected_ticker_from_dropdown == "No matches found. Try a different ticker.":
        ticker_symbol_for_analysis = None
    else:
        # Prioritize exact match from text input if it's a valid ticker
        if search_query and search_query in available_tickers and search_query == selected_ticker_from_dropdown:
            ticker_symbol_for_analysis = search_query
        else:
            ticker_symbol_for_analysis = selected_ticker_from_dropdown
elif search_query and search_query in available_tickers:
    # If user typed an exact valid ticker and didn't interact with dropdown
    ticker_symbol_for_analysis = search_query
else:
    ticker_symbol_for_analysis = None

# Display warning if no valid ticker is selected
if not ticker_symbol_for_analysis:
    st.sidebar.warning("Please enter a valid ticker or select from suggestions to proceed.")


# --- Date Range Input ---
# Use columns for side-by-side date inputs
col_start_date, col_end_date = st.sidebar.columns(2)

with col_end_date:
    end_date = st.date_input(
        "End Date", 
        datetime.now().date(), 
        max_value=datetime.now().date(), # Restrict end date to today or past
        key="end_date_input"
    ) # .date() to get just date part

with col_start_date:
    start_date = st.date_input(
        "Start Date", 
        end_date - timedelta(days=5 * 365), # 5 years of historical data default
        key="start_date_input"
    ) # .date() to get just date part


# Ensure start_date is before end_date
if start_date >= end_date:
    st.sidebar.error("Error: End date must be after start date.")
    # If dates are invalid, prevent analysis from running
    ticker_symbol_for_analysis = None # Temporarily invalidate ticker
    st.stop() # Stop execution to prevent further errors

forecast_days = st.sidebar.slider("Days to Forecast", 7, 365, 30, key="forecast_days_slider")

# No "Run Analysis" button here anymore
st.sidebar.markdown('</div>', unsafe_allow_html=True) # Close the sidebar-input-box div


# --- Sidebar Navigation (Dropdown) ---
st.sidebar.subheader("Navigation")

# Options for navigation with Font Awesome icons and labels
nav_options_data = {
    "Market Trends": "📈 Market Trends",
    "Historical Data": "⏱️ Historical Data",
    "Forecast": "📊 Forecast",
    "Metrics": "📈 Metrics",
    "News Sentiment": "📰 News Sentiment",
    "Chatbot": "💬 Chatbot" # New Chatbot tab
}

# Get the current selected page from session state, or default
current_selection_label = nav_options_data.get(st.session_state.selected_page_id, "📈 Market Trends")

# Create the selectbox
selected_page_label = st.sidebar.selectbox(
    "Go to:",
    options=list(nav_options_data.values()),
    index=list(nav_options_data.values()).index(current_selection_label),
    key="sidebar_navigation_selectbox",
    on_change=lambda: st.session_state.update(
        selected_page_id=[k for k, v in nav_options_data.items() if v == st.session_state.sidebar_navigation_selectbox][0]
    )
)
st.sidebar.markdown('</div>', unsafe_allow_html=True) # Close sidebar-nav-box

# After all UI elements are rendered, check if an analysis needs to be run
# This needs to be outside the sidebar container, but after inputs are gathered
if st.session_state.selected_page_id in ["Historical Data", "Forecast", "Metrics", "News Sentiment", "Chatbot"]: # Include Chatbot
    run_full_analysis(
        ticker_symbol_for_analysis, 
        start_date.strftime('%Y-%m-%d'), 
        end_date.strftime('%Y-%m-%d'), 
        forecast_days
    )

st.sidebar.markdown("---") # Separator between navigation and input parameters


# --- Display content based on selected sidebar page ---

if st.session_state.selected_page_id == "Market Trends":
    st.subheader("Current Market Trends")
    # Create a container for the trends to apply consistent styling
    st.markdown('<div style="border: 1px solid var(--border-color); border-radius: 8px; padding: 10px; margin-bottom: 20px; background-color: var(--secondary-background-color);">', unsafe_allow_html=True)

    with st.spinner("Loading current market trends..."):
        for item in MARKET_WATCH_ITEMS:
            ticker = item['ticker']
            name = item['name']
            
            # Fetch data for sparkline (e.g., last 30 days) and current value/change
            df_trend = load_data(ticker, (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d'))
            
            if not df_trend.empty:
                current_price = df_trend['Close'].iloc[-1]
                # Calculate percentage change from previous day
                if len(df_trend) >= 2:
                    previous_close = df_trend['Close'].iloc[-2]
                    percentage_change = ((current_price - previous_close) / previous_close) * 100
                else:
                    percentage_change = 0.0 # Not enough data for change

                color = "green" if percentage_change >= 0 else "red"
                change_sign = "+" if percentage_change >= 0 else ""

                # Using st.columns for better layout control
                col_name, col_sparkline, col_price_change = st.columns([2, 3, 2])
                with col_name:
                    st.markdown(f'<span style="font-weight: bold; font-size: 1.1em;">{name}</span><br><span style="font-size: 0.8em; color: var(--text-color);">{ticker}</span>', unsafe_allow_html=True)
                
                with col_sparkline:
                    fig_sparkline = go.Figure(data=go.Scatter(x=df_trend['Date'], y=df_trend['Close'], mode='lines', line=dict(color=color, width=1)))
                    fig_sparkline.update_layout(
                        height=50, 
                        margin=dict(l=0, r=0, t=0, b=0), 
                        xaxis=dict(visible=False, showgrid=False), 
                        yaxis=dict(visible=False, showgrid=False),
                        plot_bgcolor='rgba(0,0,0,0)', # Transparent background
                        paper_bgcolor='rgba(0,0,0,0)' # Transparent background
                    )
                    st.plotly_chart(fig_sparkline, use_container_width=True, config={'displayModeBar': False})
                
                with col_price_change:
                    st.markdown(f'<div style="text-align: right;"><span style="font-weight: bold; font-size: 1.1em;">{current_price:,.2f}</span><br><span style="color: {color}; font-size: 0.9em;">{change_sign}{percentage_change:.2f}%</span></div>', unsafe_allow_html=True)
            else:
                # Using st.columns for better layout control even when data is missing
                col_name, col_sparkline, col_price_change = st.columns([2, 3, 2])
                with col_name:
                    st.markdown(f'<span style="font-weight: bold; font-size: 1.1em;">{name}</span><br><span style="font-size: 0.8em; color: var(--text-color);">{ticker}</span>', unsafe_allow_html=True)
                with col_sparkline:
                    st.markdown('<div style="color: var(--text-color); text-align: center;">Data not available.</div>', unsafe_allow_html=True)
                with col_price_change:
                    st.write("") # Empty column for alignment
            
            # Add a horizontal line separator for visual clarity between items
            st.markdown("---") 

    st.markdown('</div>', unsafe_allow_html=True) # Close container

elif st.session_state.selected_page_id in ["Historical Data", "Forecast", "Metrics", "News Sentiment"]:
    # Check if analysis has been run for the current parameters
    # Retrieve current values of inputs to compare with analysis_triggered_params
    current_input_params = (
        ticker_symbol_for_analysis, 
        start_date.strftime('%Y-%m-%d'), 
        end_date.strftime('%Y-%m-%d'), 
        forecast_days
    )

    if st.session_state.analysis_triggered_params == current_input_params and st.session_state.current_df is not None:
        
        if st.session_state.selected_page_id == "Historical Data":
            df = st.session_state.current_df
            ticker_symbol = st.session_state.current_ticker

            st.subheader(f"Historical Data for {ticker_symbol}")
            # Display all OHLCV data
            st.dataframe(df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].tail(10)) # Display last 10 rows

            # Plot raw data
            fig_raw = go.Figure()
            fig_raw.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close Price'))
            fig_raw.add_trace(go.Scatter(x=df['Date'], y=df['Open'], mode='lines', name='Open Price'))
            fig_raw.add_trace(go.Scatter(x=df['Date'], y=df['High'], mode='lines', name='High Price'))
            fig_raw.add_trace(go.Scatter(x=df['Date'], y=df['Low'], mode='lines', name='Low Price'))
            fig_raw.update_layout(
                title_text=f'{ticker_symbol} Stock Price History (OHLC)', 
                xaxis_rangeslider_visible=True,
                hovermode='x unified'
            )
            st.plotly_chart(fig_raw, use_container_width=True)

        elif st.session_state.selected_page_id == "Forecast":
            ticker_symbol = st.session_state.current_ticker
            forecast_days = st.session_state.current_forecast_days
            prophet_forecast = st.session_state.prophet_forecast
            xgb_future_predictions = st.session_state.xgb_future_predictions
            ensemble_df = st.session_state.ensemble_df
            df = st.session_state.current_df # Original historical data

            st.markdown("### Model Training and Forecasting")

            # Create two columns for the forecast and indicators
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Forecast Visualization")

                fig_forecast = go.Figure()

                # Filter Prophet forecast to only include the future forecast days
                prophet_future_only = prophet_forecast.iloc[-forecast_days:].copy() 

                # Plot Prophet Forecast (future only)
                fig_forecast.add_trace(go.Scatter(
                    x=prophet_future_only['ds'],
                    y=prophet_future_only['yhat'],
                    mode='lines',
                    name='Prophet Forecast',
                    line=dict(color='blue', dash='dash')
                ))
                # Prophet Confidence Interval (future only)
                fig_forecast.add_trace(go.Scatter(
                    x=prophet_future_only['ds'],
                    y=prophet_future_only['yhat_lower'],
                    fill='tonexty',
                    fillcolor='rgba(0,0,255,0.1)', # Light blue for Prophet CI
                    line=dict(color='rgba(0,0,0,0)'),
                    showlegend=False,
                    name='Prophet Confidence Interval Lower'
                ))
                fig_forecast.add_trace(go.Scatter(
                    x=prophet_future_only['ds'],
                    y=prophet_future_only['yhat_upper'],
                    fill='tonexty',
                    fillcolor='rgba(0,0,255,0.1)', # Light blue for Prophet CI
                    line=dict(color='rgba(0,0,0,0)'),
                    showlegend=False,
                    name='Prophet Confidence Interval Upper'
                ))


                # Plot XGBoost Forecast
                if xgb_future_predictions.size > 0:
                    # Create dates for XGBoost future predictions
                    last_historical_date = df['Date'].max()
                    # Ensure xgb_forecast_dates matches the length of xgb_future_predictions
                    xgb_forecast_dates = [last_historical_date + timedelta(days=i) for i in range(1, len(xgb_future_predictions) + 1)]

                    fig_forecast.add_trace(go.Scatter(
                        x=xgb_forecast_dates,
                        y=xgb_future_predictions.flatten(),
                        mode='lines',
                        name='XGBoost Forecast',
                        line=dict(color='green', dash='dot')
                    ))
                else:
                    st.warning("XGBoost future predictions are not available for plotting.")

                # Plot Ensemble Forecast
                fig_forecast.add_trace(go.Scatter(
                    x=ensemble_df['Date'],
                    y=ensemble_df['Ensemble Forecast'],
                    mode='lines',
                    name='Ensemble Forecast',
                    line=dict(color='purple', width=2)
                ))

                fig_forecast.update_layout(
                    title=f'{ticker_symbol} Stock Price Forecast (Next {forecast_days} Days)',
                    xaxis_title='Date',
                    yaxis_title='Close Price',
                    xaxis_rangeslider_visible=True, 
                    hovermode='x unified',
                    height=400 # Adjusted height for better balance
                )
                st.plotly_chart(fig_forecast, use_container_width=True)

                st.subheader("Forecast Data (Ensemble)")
                st.dataframe(ensemble_df) # Displaying ensemble forecast data

            with col2:
                st.markdown("### Technical Indicators")

                # Determine the start date for historical context in indicator plots (e.g., last 180 days)
                indicator_context_days = 180
                start_date_for_indicators_plot = df['Date'].max() - timedelta(days=indicator_context_days)
                df_indicators_context = df[df['Date'] >= start_date_for_indicators_plot].copy()

                # Calculate and plot 20-day Simple Moving Average (SMA)
                df_indicators_context['SMA_20'] = df_indicators_context['Close'].rolling(window=20).mean()
                fig_sma = go.Figure()
                fig_sma.add_trace(go.Scatter(
                    x=df_indicators_context['Date'],
                    y=df_indicators_context['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='lightgray', width=1)
                ))
                fig_sma.add_trace(go.Scatter(
                    x=df_indicators_context['Date'],
                    y=df_indicators_context['SMA_20'],
                    mode='lines',
                    name='20-day SMA',
                    line=dict(color='orange', width=2)
                ))
                fig_sma.update_layout(
                    title=f'{ticker_symbol} 20-day Simple Moving Average (Last {indicator_context_days} Days)',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    xaxis_rangeslider_visible=True,
                    hovermode='x unified',
                    height=400 # Consistent height
                )
                st.plotly_chart(fig_sma, use_container_width=True)

                # Calculate and plot Bollinger Bands
                df_indicators_context['BB_Middle'] = df_indicators_context['Close'].rolling(window=20).mean()
                df_indicators_context['BB_StdDev'] = df_indicators_context['Close'].rolling(window=20).std()
                df_indicators_context['BB_Upper'] = df_indicators_context['BB_Middle'] + (df_indicators_context['BB_StdDev'] * 2)
                df_indicators_context['BB_Lower'] = df_indicators_context['BB_Middle'] - (df_indicators_context['BB_StdDev'] * 2)

                fig_bb = go.Figure()
                fig_bb.add_trace(go.Scatter(
                    x=df_indicators_context['Date'],
                    y=df_indicators_context['BB_Upper'],
                    mode='lines',
                    name='Bollinger Upper',
                    line=dict(color='red', dash='dot', width=1)
                ))
                fig_bb.add_trace(go.Scatter(
                    x=df_indicators_context['Date'],
                    y=df_indicators_context['BB_Lower'],
                    mode='lines',
                    name='Bollinger Lower',
                    line=dict(color='red', dash='dot', width=1)
                ))
                fig_bb.add_trace(go.Scatter(
                    x=df_indicators_context['Date'],
                    y=df_indicators_context['BB_Middle'],
                    mode='lines',
                    name='Bollinger Middle (20-day SMA)',
                    line=dict(color='purple', width=1)
                ))
                fig_bb.update_layout(
                    title=f'{ticker_symbol} Bollinger Bands (Last {indicator_context_days} Days)',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    xaxis_rangeslider_visible=True,
                    hovermode='x unified',
                    height=400 # Consistent height
                )
                st.plotly_chart(fig_bb, use_container_width=True)

        elif st.session_state.selected_page_id == "Metrics":
            st.markdown("### Model Evaluation")
            # Check for necessary variables before calling evaluate_models
            if (st.session_state.predictions_xgb is not None and 
                st.session_state.y_test_xgb is not None and 
                st.session_state.scaler is not None and 
                st.session_state.train_size_xgb is not None and 
                st.session_state.LOOK_BACK is not None):

                metrics_data = evaluate_models(
                    st.session_state.current_df.copy(), # Pass a copy to avoid modifying original
                    st.session_state.prophet_forecast, 
                    st.session_state.predictions_xgb, 
                    st.session_state.y_test_xgb, 
                    st.session_state.scaler, 
                    st.session_state.train_size_xgb, 
                    st.session_state.LOOK_BACK
                )
                
                # Filter metrics for Prophet and display in a separate table
                prophet_metrics = [m for m in metrics_data if m["Model"] == "Prophet"]
                st.subheader("Prophet Model Metrics")
                st.dataframe(pd.DataFrame(prophet_metrics))

                # Filter metrics for XGBoost and display in a separate table
                xgboost_metrics = [m for m in metrics_data if m["Model"] == "XGBoost"]
                st.subheader("XGBoost Model Metrics")
                st.dataframe(pd.DataFrame(xgboost_metrics))

            else:
                st.info("Model evaluation metrics are not available. Please ensure data is loaded and models are trained, and that sufficient historical data is provided (at least 60 days recommended).")

        elif st.session_state.selected_page_id == "News Sentiment":
            st.markdown("### Latest News Sentiment Analysis")

            # Callback function for the "Show News Sentiment" button
            def on_show_news_click_and_fetch(current_ticker_for_news, api_key_for_news):
                if current_ticker_for_news:
                    with st.spinner(f"Fetching and analyzing news for {current_ticker_for_news}..."):
                        news_articles, overall_summary = fetch_news(current_ticker_for_news, api_key_for_news)
                        st.session_state.news_articles_data = news_articles
                        st.session_state.overall_sentiment_summary = overall_summary
                        st.session_state.last_news_ticker_fetched = current_ticker_for_news
                else:
                    st.session_state.news_articles_data = None
                    st.session_state.overall_sentiment_summary = None
                    st.session_state.last_news_ticker_fetched = None

            # The button to trigger news fetching
            # Enable the news button only if a valid ticker is present
            if ticker_symbol_for_analysis:
                if st.button("Show News Sentiment", key="show_news_button", 
                            on_click=lambda: on_show_news_click_and_fetch(ticker_symbol_for_analysis, FINNHUB_API_KEY)): # Changed EODHD_API_KEY to FINNHUB_API_KEY
                    pass # The action is handled by the on_click callback
            else:
                st.info("Please select a stock ticker from the sidebar to enable news sentiment analysis.")


            # Display news only if it's in session state and belongs to the current ticker
            if st.session_state.news_articles_data is not None and st.session_state.last_news_ticker_fetched == ticker_symbol_for_analysis:
                news_articles = st.session_state.news_articles_data
                overall_summary = st.session_state.overall_sentiment_summary

                if overall_summary:
                    st.markdown("#### Overall News Sentiment Summary")
                    st.markdown(f"""
                        <div class="overall-summary-box">
                            <p>Total Articles: <strong>{overall_summary['total_articles']}</strong></p>
                            <p>Positive: <span class="positive-count">{overall_summary['positive_count']}</span></p>
                            <p>Negative: <span class="negative-count">{overall_summary['negative_count']}</span></p>
                            <p>Neutral: <span class="neutral-count">{overall_summary['neutral_count']}</span></p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.markdown("---")

                if news_articles:
                    st.markdown("#### Recent News Articles and Sentiment")
                    # Display only the first 10 articles
                    for i, article in enumerate(news_articles[:10]): 
                        title = article.get('title', 'No Title')
                        content = article.get('content', title)
                        url = article.get('link', '#')

                        sentiment_label = article.get('sentiment', 'neutral').lower()
                        sentiment_score = article.get('score', 0.5)

                        # Apply color_class and emoji based on sentiment
                        color_class = ""
                        emoji = ""
                        if sentiment_label == "positive":
                            color_class = "sentiment-positive"
                            emoji = "😊"
                        elif sentiment_label == "negative":
                            color_class = "sentiment-negative"
                            emoji = "😞"
                        else: # neutral
                            color_class = "sentiment-neutral"
                            emoji = "😐" # Neutral emoji

                        st.markdown(f"""
                            <div class="news-article-container">
                                <h5>{title}</h5>
                                <p>{content[:200]}...</p>
                                <p class="{color_class}">Sentiment: {sentiment_label.capitalize()} {emoji} (Score: {sentiment_score:.2f})</p>
                                <a href="{url}" target="_blank">Read More</a>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Create a unique key for the summarization button and output for each article
                        summarize_button_key = f"summarize_article_{i}"
                        
                        # Check if summary is already in cache
                        summary_cache_key = f"summary_{i}_{hash(content)}" # Unique key for article content

                        # Conditionally display the summarize button and summary output
                        if st.button(f"✨ Summarize Article {i+1}", key=summarize_button_key):
                            if summary_cache_key in st.session_state.summaries_cache:
                                # If summary exists, hide/show it
                                if st.session_state.get(f"show_summary_{i}", False):
                                    st.session_state[f"show_summary_{i}"] = False
                                else:
                                    st.session_state[f"show_summary_{i}"] = True
                            else:
                                # If summary doesn't exist, generate it
                                with st.spinner(f"Summarizing article {i+1} using Gemini..."):
                                    try:
                                        summary = summarize_article_gemini(content)
                                        st.session_state.summaries_cache[summary_cache_key] = summary # Cache the summary
                                        st.session_state[f"show_summary_{i}"] = True # Show after generation
                                    except Exception as e:
                                        # Store the detailed error message in the cache
                                        st.session_state.summaries_cache[summary_cache_key] = f"Error summarizing article: {e}"
                                        st.session_state[f"show_summary_{i}"] = True # Show error message
                                        st.error(f"Failed to summarize article {i+1}: {e}") # Keep this for immediate visibility

                            # Rerun to update the display (show/hide summary, or show new summary/error)
                            st.rerun() 
                        
                        # Display the summary if it's in cache and marked to be shown
                        if summary_cache_key in st.session_state.summaries_cache and st.session_state.get(f"show_summary_{i}", False):
                            st.markdown(f'<div class="summary-output"><strong>Summary:</strong> {st.session_state.summaries_cache[summary_cache_key]}</div>', unsafe_allow_html=True)

                else:
                    st.write("No news articles found for the given ticker or API key is invalid.")
            # This block provides a message if a ticker is selected but the news button hasn't been clicked yet.
            elif ticker_symbol_for_analysis: 
                st.info("Click 'Show News Sentiment' to fetch and analyze news for the selected ticker.")
            # This block provides a message if no ticker is selected at all (initial state).
            else: 
                st.info("Select a stock ticker from the sidebar to enable news sentiment.")
    else:
        st.info("Please select a stock ticker and adjust the parameters in the sidebar. The analysis will run automatically when you select a relevant tab like 'Historical Data', 'Forecast', 'Metrics', or 'News Sentiment'.")

elif st.session_state.selected_page_id == "Chatbot":
    st.subheader("Stock Analysis Chatbot 💬")
    st.markdown("Ask me anything about the selected stock's historical data, forecasts, or news sentiment!")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input for user query
    user_query = st.chat_input("Ask me about the stock...")

    if user_query:
        # Add user query to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.spinner("Thinking..."):
            # Generate response from Gemini
            # Pass all necessary context data from session state
            gemini_response = generate_chatbot_response(
                user_query,
                st.session_state.current_ticker,
                st.session_state.current_df,
                st.session_state.ensemble_df,
                st.session_state.news_articles_data,
                st.session_state.overall_sentiment_summary,
                st.session_state.chat_history # Pass the entire chat history for context
            )
            
            # Add Gemini's response to chat history
            st.session_state.chat_history.append({"role": "model", "content": gemini_response})
            with st.chat_message("model"):
                st.markdown(gemini_response)
    
    # --- Suggestion Questions ---
    st.markdown("---")
    st.markdown("#### Suggested Questions:")
    
    suggestion_queries = []
    if st.session_state.current_ticker:
        ticker_name = st.session_state.current_ticker
        
        # General questions
        suggestion_queries.append(f"What is the current price of {ticker_name}?")
        suggestion_queries.append(f"Summarize the recent news sentiment for {ticker_name}.")
        
        # Forecasting questions (if forecast data is available)
        if st.session_state.ensemble_df is not None and not st.session_state.ensemble_df.empty:
            suggestion_queries.append(f"What is the forecast for {ticker_name} for the next {st.session_state.current_forecast_days} days?")
            suggestion_queries.append(f"How accurate are the forecasting models for {ticker_name}?")
        
        # News sentiment questions (if news data is available)
        if st.session_state.news_articles_data:
            suggestion_queries.append(f"Tell me about the most positive news for {ticker_name}.")
            suggestion_queries.append(f"Are there any negative news articles for {ticker_name}?")
        
        # Historical data questions (if historical data is available)
        if st.session_state.current_df is not None and not st.session_state.current_df.empty:
            suggestion_queries.append(f"Show me the historical closing prices for {ticker_name} for the last month.")
            suggestion_queries.append(f"What are the technical indicators for {ticker_name}?")

    else:
        suggestion_queries.append("Please select a stock ticker from the sidebar first to enable specific questions.")
        suggestion_queries.append("What is this app about?")
    
    # Display suggestion buttons (limited to 4)
    cols = st.columns(2)
    for i, query in enumerate(suggestion_queries[:4]): # Slice to get only the first 4 suggestions
        if i % 2 == 0:
            with cols[0]:
                if st.button(query, key=f"suggest_q_{i}"):
                    st.session_state.chat_history.append({"role": "user", "content": query})
                    with st.chat_message("user"):
                        st.markdown(query)
                    with st.spinner("Thinking..."):
                        # Pass all context data to generate_chatbot_response
                        gemini_response = generate_chatbot_response(
                            query,
                            st.session_state.current_ticker,
                            st.session_state.current_df,
                            st.session_state.ensemble_df,
                            st.session_state.news_articles_data,
                            st.session_state.overall_sentiment_summary,
                            st.session_state.chat_history # Pass the entire chat history for context
                        )
                        st.session_state.chat_history.append({"role": "model", "content": gemini_response})
                        with st.chat_message("model"):
                            st.markdown(gemini_response)
                    st.rerun() # Rerun to update chat history and clear input
        else:
            with cols[1]:
                if st.button(query, key=f"suggest_q_{i}"):
                    st.session_state.chat_history.append({"role": "user", "content": query})
                    with st.chat_message("user"):
                        st.markdown(query)
                    with st.spinner("Thinking..."):
                        # Pass all context data to generate_chatbot_response
                        gemini_response = generate_chatbot_response(
                            query,
                            st.session_state.current_ticker,
                            st.session_state.current_df,
                            st.session_state.ensemble_df,
                            st.session_state.news_articles_data,
                            st.session_state.overall_sentiment_summary,
                            st.session_state.chat_history # Pass the entire chat history for context
                        )
                        st.session_state.chat_history.append({"role": "model", "content": gemini_response})
                        with st.chat_message("model"):
                            st.markdown(gemini_response)
                    st.rerun() # Rerun to update chat history and clear input
