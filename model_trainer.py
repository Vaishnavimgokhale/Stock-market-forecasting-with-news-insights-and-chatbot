import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import timedelta
import streamlit as st

# Global constant for look-back period for XGBoost features
LOOK_BACK = 10

def create_features_and_target(df, look_back=LOOK_BACK):
    """
    Creates lagged features and time-based features for XGBoost model.
    'Date' column is assumed to be the datetime index.
    'Close' column is assumed to be the target variable.
    Returns features and target in their ORIGINAL scale.
    """
    df_copy = df.copy()
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    df_copy.set_index('Date', inplace=True)

    # Create lagged features based on 'Close' price
    for i in range(1, look_back + 1):
        df_copy[f'Lag_{i}'] = df_copy['Close'].shift(i)

    # Create time-based features
    df_copy['DayOfWeek'] = df_copy.index.dayofweek
    df_copy['DayOfMonth'] = df_copy.index.day
    df_copy['Month'] = df_copy.index.month
    df_copy['Year'] = df_copy.index.year
    df_copy['WeekOfYear'] = df_copy.index.isocalendar().week 

    # Define the exact features for X that will be used for training and prediction
    features_for_X = [f'Lag_{i}' for i in range(1, look_back + 1)] + \
                     ['DayOfWeek', 'DayOfMonth', 'Month', 'Year', 'WeekOfYear']

    # Select only these features and the target 'Close' before dropping NaNs
    df_processed = df_copy[features_for_X + ['Close']].dropna()

    # Define features (X) and target (y) from the processed DataFrame
    X = df_processed[features_for_X]
    y = df_processed['Close']

    return X, y

def train_prophet_model(df):
    """
    Trains a Prophet model.
    Expects DataFrame with 'Date' and 'Close' columns.
    """
    prophet_df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(prophet_df)
    return model, prophet_df

def predict_prophet(model, prophet_df, forecast_days):
    """
    Generates future predictions using a trained Prophet model.
    """
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    return forecast

def train_xgboost_model(X_train, y_train):
    """
    Trains an XGBoost Regressor model.
    """
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05,
                             max_depth=5, subsample=0.7, colsample_bytree=0.7, random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_xgboost(model, X_future):
    """
    Generates future predictions using a trained XGBoost model.
    """
    predictions = model.predict(X_future)
    return predictions

def combine_predictions(prophet_forecast, xgb_future_predictions, original_df, forecast_days, scaler):
    """
    Combines predictions from Prophet and XGBoost using equal weighting.
    Inverse transforms scaled predictions.
    """
    # Get the last historical date from the original DataFrame
    last_historical_date = original_df['Date'].max()

    # Create future dates for the ensemble DataFrame
    future_dates = [last_historical_date + timedelta(days=i) for i in range(1, forecast_days + 1)]

    # Ensure all predictions are aligned to the future_dates
    # Prophet forecast already has 'ds' which are the future dates
    prophet_future_df = prophet_forecast.iloc[-forecast_days:].copy()
    
    # Prophet predictions are already in original scale
    prophet_future_predictions_original_scale = prophet_future_df['yhat'].values.flatten()

    # XGBoost predictions are already in original scale if inverse_transform was applied before passing
    # Assuming xgb_future_predictions is already inverse-transformed
    xgb_predictions_original_scale = xgb_future_predictions.flatten()


    # Create a DataFrame for combined predictions
    ensemble_df = pd.DataFrame({'Date': future_dates})
    ensemble_df['Prophet Forecast'] = prophet_future_predictions_original_scale[:forecast_days]
    
    # Handle cases where xgb_predictions_original_scale might be shorter than forecast_days
    # This can happen if XGBoost couldn't generate full future predictions due to data issues
    if len(xgb_predictions_original_scale) < forecast_days:
        # Pad with NaNs or zeros, or truncate other forecasts
        # For now, we'll pad with NaN and then drop NaNs in ensemble calculation
        padded_xgb_predictions = np.pad(xgb_predictions_original_scale, 
                                        (0, max(0, forecast_days - len(xgb_predictions_original_scale))), 
                                        'constant', constant_values=np.nan)
        ensemble_df['XGBoost Forecast'] = padded_xgb_predictions
    else:
        ensemble_df['XGBoost Forecast'] = xgb_predictions_original_scale[:forecast_days]


    # Simple average ensemble (now only two models)
    # Use .mean(axis=1) and dropna to handle potential NaNs from padded XGBoost forecasts
    ensemble_df['Ensemble Forecast'] = ensemble_df[['Prophet Forecast', 'XGBoost Forecast']].mean(axis=1)

    return ensemble_df


def evaluate_models(original_df, prophet_forecast, predictions_xgb, y_test_xgb, scaler, train_size_xgb, LOOK_BACK):
    """
    Evaluates Prophet and XGBoost models and returns metrics in a structured format.
    """
    metrics_data = []

    # Prophet Evaluation
    # Align Prophet predictions with actual historical data for evaluation
    prophet_historical_predictions = prophet_forecast['yhat'].iloc[:len(original_df)].values
    y_actual_prophet = original_df['Close'].values

    # Ensure arrays have the same length for RMSE, MAE, R2
    min_len = min(len(y_actual_prophet), len(prophet_historical_predictions))
    prophet_rmse = np.sqrt(mean_squared_error(y_actual_prophet[:min_len], prophet_historical_predictions[:min_len]))
    prophet_mae = mean_absolute_error(y_actual_prophet[:min_len], prophet_historical_predictions[:min_len])
    prophet_r2 = r2_score(y_actual_prophet[:min_len], prophet_historical_predictions[:min_len])

    metrics_data.append({
        "Model": "Prophet",
        "Metric": "RMSE",
        "Value": f"{prophet_rmse:.2f}"
    })
    metrics_data.append({
        "Model": "Prophet",
        "Metric": "MAE",
        "Value": f"{prophet_mae:.2f}"
    })
    metrics_data.append({
        "Model": "Prophet",
        "Metric": "R-squared",
        "Value": f"{prophet_r2:.2f}"
    })

    # XGBoost Evaluation (on test set)
    # Check if test data and predictions exist and are not empty
    if len(y_test_xgb) > 0 and len(predictions_xgb) > 0:
        xgb_rmse = np.sqrt(mean_squared_error(y_test_xgb, predictions_xgb))
        xgb_mae = mean_absolute_error(y_test_xgb, predictions_xgb)
        xgb_r2 = r2_score(y_test_xgb, predictions_xgb)

        metrics_data.append({
            "Model": "XGBoost",
            "Metric": "RMSE",
            "Value": f"{xgb_rmse:.2f}"
        })
        metrics_data.append({
            "Model": "XGBoost",
            "Metric": "MAE",
            "Value": f"{xgb_mae:.2f}"
        })
        metrics_data.append({
            "Model": "XGBoost",
            "Metric": "R-squared",
            "Value": f"{xgb_r2:.2f}"
        })
    else:
        metrics_data.append({
            "Model": "XGBoost",
            "Metric": "Status",
            "Value": "Skipped (Insufficient Test Data)"
        })
    
    return metrics_data


def train_and_forecast_models(df, forecast_days):
    """
    Orchestrates training and forecasting for Prophet and XGBoost.
    Returns all necessary data for plotting and evaluation.
    """
    # --- 1. Data Preprocessing for all models ---
    # Prophet uses 'ds' and 'y'
    prophet_df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

    # Fit the scaler on the entire 'Close' column from the original DataFrame
    data_for_scaling = df[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data_for_scaling) # Fit scaler here once

    # Create features (X) and target (y) in their original scales
    X_original_scale, y_original_scale = create_features_and_target(df) 

    # Now, scale the features (Lag_X) for XGBoost training
    # Time-based features do not need scaling
    xgboost_feature_columns = [f'Lag_{j}' for j in range(1, LOOK_BACK + 1)] + \
                                  ['DayOfWeek', 'DayOfMonth', 'Month', 'Year', 'WeekOfYear']
    
    # Create a DataFrame for scaled features for XGBoost
    X_scaled_for_xgb = pd.DataFrame(index=X_original_scale.index)
    
    # Scale lagged features
    for j in range(1, LOOK_BACK + 1):
        # Clip original values to scaler's range before transforming
        clipped_lag_values = np.clip(X_original_scale[f'Lag_{j}'].values, scaler.data_min_[0], scaler.data_max_[0])
        X_scaled_for_xgb[f'Lag_{j}'] = scaler.transform(clipped_lag_values.reshape(-1, 1)).flatten()

    # Add time-based features without scaling
    for col in ['DayOfWeek', 'DayOfMonth', 'Month', 'Year', 'WeekOfYear']:
        X_scaled_for_xgb[col] = X_original_scale[col]

    # Scale the target 'y' for XGBoost training
    y_scaled_for_xgb = scaler.transform(y_original_scale.values.reshape(-1, 1)).flatten()

    # Split data for XGBoost (using the scaled features and target)
    train_size_xgb = int(len(X_scaled_for_xgb) * 0.8)
    X_train_xgb, X_test_xgb = X_scaled_for_xgb[:train_size_xgb], X_scaled_for_xgb[train_size_xgb:]
    y_train_scaled_xgb, y_test_scaled_xgb = y_scaled_for_xgb[:train_size_xgb], y_scaled_for_xgb[train_size_xgb:]
    
    # Keep y_test_original_scale for evaluation metrics
    y_test_original_scale = y_original_scale.iloc[train_size_xgb:] 

    # Initialize XGBoost specific outputs to handle cases where it might not run
    xgb_future_predictions_original_scale = np.array([])
    predictions_xgb_original_scale = np.array([])

    # --- 2. Prophet Model ---
    prophet_model, _ = train_prophet_model(df) # Pass original df, Prophet handles its own 'ds', 'y'
    prophet_forecast = predict_prophet(prophet_model, prophet_df, forecast_days)

    # --- 3. XGBoost Model (Conditional Training and Prediction) ---
    # Ensure there's enough data for both training and testing for XGBoost
    if not X_train_xgb.empty and not X_test_xgb.empty and len(y_train_scaled_xgb) > 0 and len(y_test_scaled_xgb) > 0:
        xgboost_model = train_xgboost_model(X_train_xgb, y_train_scaled_xgb)

        # Prepare future features for XGBoost
        last_known_data = df.tail(LOOK_BACK)
        future_dates = [df['Date'].max() + timedelta(days=i) for i in range(1, forecast_days + 1)]
        
        # List to collect dictionaries of future features
        future_features_list = [] 
        
        # Populate future_X_xgb with lagged values and time features
        current_lags_original_scale = last_known_data['Close'].values[::-1].tolist() # Reverse to get most recent first
        
        for i in range(forecast_days):
            new_row_features = {}
            # Lag features: use current_lags, shifting as new predictions come in
            for j in range(1, LOOK_BACK + 1):
                if j-1 < len(current_lags_original_scale):
                    # Clip original values to scaler's range before transforming
                    clipped_original_lag = np.clip(current_lags_original_scale[j-1], scaler.data_min_[0], scaler.data_max_[0])
                    scaled_lag = scaler.transform(np.array(clipped_original_lag).reshape(-1, 1)).flatten()[0]
                    new_row_features[f'Lag_{j}'] = scaled_lag
                else:
                    new_row_features[f'Lag_{j}'] = np.nan

            # Time features for future dates
            future_date = future_dates[i]
            new_row_features['DayOfWeek'] = future_date.dayofweek
            new_row_features['DayOfMonth'] = future_date.day
            new_row_features['Month'] = future_date.month
            new_row_features['Year'] = future_date.year
            new_row_features['WeekOfYear'] = future_date.isocalendar().week 

            # Append the dictionary of features for this future date
            future_features_list.append(new_row_features)
            
            # Predict the next value using the current set of features
            # Ensure the DataFrame for prediction has the correct columns and order
            pred_df_for_next_step = pd.DataFrame([new_row_features], columns=xgboost_feature_columns)
            next_prediction_scaled = xgboost_model.predict(pred_df_for_next_step)
            
            # Clip the scaled prediction to ensure it's within [0, 1] before inverse transforming
            clipped_next_prediction_scaled = np.clip(next_prediction_scaled, 0, 1)

            # Update current_lags_original_scale with the new prediction (inverse transformed) for the next iteration
            current_lags_original_scale.insert(0, scaler.inverse_transform(clipped_next_prediction_scaled.reshape(-1, 1)).flatten()[0]) # Add new prediction to front
            current_lags_original_scale = current_lags_original_scale[:LOOK_BACK] # Keep only the required number of lags

        # After the loop, create the full future_X_xgb DataFrame from the collected features
        if future_features_list:
            future_X_xgb = pd.DataFrame(future_features_list, columns=xgboost_feature_columns)
        else:
            future_X_xgb = pd.DataFrame(columns=xgboost_feature_columns) # Empty DataFrame if no rows

        # Predict on future features (all at once after loop)
        xgb_future_predictions_scaled = xgboost_model.predict(future_X_xgb)
        # Clip the final batch of scaled predictions before inverse transforming
        clipped_xgb_future_predictions_scaled = np.clip(xgb_future_predictions_scaled, 0, 1)
        xgb_future_predictions_original_scale = scaler.inverse_transform(clipped_xgb_future_predictions_scaled.reshape(-1, 1))

        # Predict on test set for evaluation
        predictions_xgb_scaled = xgboost_model.predict(X_test_xgb)
        # Clip the test set scaled predictions before inverse transforming
        clipped_predictions_xgb_scaled = np.clip(predictions_xgb_scaled, 0, 1)
        predictions_xgb_original_scale = scaler.inverse_transform(clipped_predictions_xgb_scaled.reshape(-1, 1))
    else:
        st.warning("Not enough data for a valid XGBoost train/test split. XGBoost predictions and metrics will be skipped.")
        # Ensure these are empty arrays if no test data
        y_test_original_scale = np.array([])
        predictions_xgb_original_scale = np.array([])
        xgb_future_predictions_original_scale = np.array([]) # Also ensure this is empty if XGBoost is skipped


    # --- 4. Combine Predictions --- (now only Prophet and XGBoost)
    ensemble_df = combine_predictions(
        prophet_forecast,
        xgb_future_predictions_original_scale, # Pass the potentially empty array
        df, # Pass original df for date alignment
        forecast_days,
        scaler # Pass scaler for inverse transform within combine_predictions if needed
    )

    # Adjusted return statement
    return prophet_forecast, xgb_future_predictions_original_scale, ensemble_df, predictions_xgb_original_scale, y_test_original_scale, scaler, train_size_xgb, LOOK_BACK
