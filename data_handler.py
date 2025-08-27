import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

@st.cache_data(ttl=3600) # Cache data for 1 hour
def load_data(ticker_symbol, start_date, end_date):
    """
    Loads historical stock data from Yahoo Finance and performs initial cleaning.
    Ensures 'Date' column is present and OHLCV columns are numeric.
    Includes a fallback for ^NSEI to try ^NSEI.NS if direct fetch fails.
    Graciously handles missing OHLC for index tickers like ^NSEI.
    """
    attempted_tickers = [ticker_symbol]
    # If the primary ticker is NIFTY 50, add its alternative symbol
    # NOTE: Sometimes yfinance might not provide Open/High/Low for certain indices.
    is_index_ticker = ticker_symbol.startswith('^') or '.NS' in ticker_symbol # Simple check for indices

    if ticker_symbol == '^NSEI':
        attempted_tickers.append('^NSEI.NS') # Fallback for NIFTY 50

    data = pd.DataFrame()
    successful_ticker = None

    # Try fetching data with each ticker in the list until successful
    for current_ticker in attempted_tickers:
        try:
            data = yf.download(current_ticker, start=start_date, end=end_date)
            
            # --- DEBUGGING PRINT: See what columns yfinance returns immediately ---
            print(f"DEBUG: Columns after yf.download for {current_ticker}: {data.columns.tolist()}")
            # --- END DEBUGGING PRINT ---

            if not data.empty:
                # If data is found, use this ticker and break the loop
                successful_ticker = current_ticker
                break
        except Exception as e:
            # Log or print a warning for failed attempts, but don't stop the process
            print(f"Warning: Attempting to load data for {current_ticker} failed: {e}")
            data = pd.DataFrame() # Ensure data is empty if this attempt failed

    if data.empty:
        # If no data found after all attempts, show an error and return empty DataFrame
        st.error(f"No data found for {ticker_symbol} within the specified date range after all attempts. Please check the ticker symbol and date range.")
        return pd.DataFrame() 

    # If a successful ticker was found, use that for further processing and error messages
    ticker_symbol_for_processing = successful_ticker if successful_ticker else ticker_symbol

    data.reset_index(inplace=True)

    # --- IMPORTANT: Flatten MultiIndex columns immediately if they exist ---
    # This logic is crucial if yfinance returns multi-indexed columns (e.g., from multiple tickers)
    # For single ticker download, it usually returns flat columns, but this handles robustness.
    if isinstance(data.columns, pd.MultiIndex):
        new_columns = []
        for col_tuple in data.columns:
            if isinstance(col_tuple, tuple):
                if col_tuple[0] == 'Date':
                    new_columns.append('Date') 
                elif col_tuple[1]: 
                    # Attempt to clean the ticker part from column names for better matching
                    col_base_name = col_tuple[0]
                    col_ticker_part = col_tuple[1]
                    # Check if the ticker part matches the current successful ticker (or its base form)
                    if col_ticker_part.replace('.NS', '').upper() == ticker_symbol_for_processing.replace('^', '').replace('.NS', '').upper():
                        new_columns.append(col_base_name)
                    else:
                        new_columns.append(f"{col_base_name}_{col_ticker_part}") # Keep original if not the primary ticker
                else: 
                    # If second level of tuple is empty (e.g., ('Open', '')), use first level
                    new_columns.append(col_tuple[0]) 
            else:
                new_columns.append(col_tuple) # For already flat columns
        data.columns = new_columns
        
    # After flattening, ensure standard OHLCV names if they were suffixed.
    # This loop is crucial for standardizing column names like 'Close_GOOG' to 'Close'.
    # If the column name is already 'Open', 'Close', etc., it won't be renamed.
    for col_name in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']: # Include 'Adj Close' as it might be default
        suffixed_col_name = f'{col_name}_{ticker_symbol_for_processing}'
        if suffixed_col_name in data.columns:
            data.rename(columns={suffixed_col_name: col_name}, inplace=True)
        
        # Handle cases where ticker_symbol_for_processing might be ^NSEI.NS
        # and yfinance returned columns like 'Open_NSEI.NS'
        if '.NS' in ticker_symbol_for_processing:
            simplified_ticker = ticker_symbol_for_processing.replace('^', '').replace('.NS', '')
            suffixed_col_name_ns = f'{col_name}_{simplified_ticker}.NS'
            if suffixed_col_name_ns in data.columns:
                data.rename(columns={suffixed_col_name_ns: col_name}, inplace=True)


    # Ensure 'Date' column is present and convert to datetime
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
    else:
        st.error("Error: 'Date' column not found after all processing. This is critical.")
        return pd.DataFrame()

    # Define required columns based on whether it's an index or not.
    # Indices like ^NSEI often lack Open, High, Low, but always have Close and Volume (if trading data exists).
    core_required_cols = ['Close', 'Volume']
    
    # If it's an index, we might relax the OHLC requirement, but still prefer them if available.
    if is_index_ticker and not all(col in data.columns for col in ['Open', 'High', 'Low']):
        missing_ohl = [col for col in ['Open', 'High', 'Low'] if col not in data.columns]
        if missing_ohl:
            st.warning(f"Note: Price range (Open, High, Low) data might be missing for index ticker {ticker_symbol_for_processing}. Proceeding with available data.")
        # Ensure 'Open', 'High', 'Low' columns are added with NaN if missing, to prevent KeyError later
        for col in ['Open', 'High', 'Low']:
            if col not in data.columns:
                data[col] = np.nan # Add missing columns with NaN
        
        # Now, ensure all columns (core and potentially added NaNs) are converted to numeric
        all_cols_to_convert = core_required_cols + ['Open', 'High', 'Low']
        for col in all_cols_to_convert:
            if col in data.columns: # Check if column actually exists (e.g. if 'Open' wasn't added as NaN for some reason)
                data[col] = pd.to_numeric(data[col], errors='coerce') # Convert to numeric
            else:
                st.error(f"Critical Error: Required column '{col}' is neither present nor could be added as NaN for {ticker_symbol_for_processing}.")
                return pd.DataFrame() # This should ideally not happen if logic above works
    else: # For regular stocks, all OHLCV are strictly required
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in data.columns:
                st.error(f"The '{col}' price column is missing after all processing for {ticker_symbol_for_processing}. This is critical for non-index tickers.")
                return pd.DataFrame()
            data[col] = pd.to_numeric(data[col], errors='coerce') # Convert to numeric
    
    # Drop rows where any of the core required columns became NaN after numeric conversion
    data.dropna(subset=core_required_cols, inplace=True) 

    if data.empty:
        st.error(f"After cleaning, no valid data remains for {ticker_symbol_for_processing}. This might be due to an invalid ticker, date range with no trading, or all data being non-numeric. Please check the stock ticker or date range.")
        return pd.DataFrame()

    # Return all relevant columns: Date, Open, High, Low, Close, Volume
    # Ensure these columns exist before attempting to select them.
    final_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    # Filter to only include columns that actually exist in the DataFrame
    existing_final_cols = [col for col in final_cols if col in data.columns]

    return data[existing_final_cols].copy() # Use .copy() to avoid SettingWithCopyWarning downstream
