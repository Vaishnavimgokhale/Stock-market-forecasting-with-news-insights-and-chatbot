# Stock-market-forecasting-with-news-insights-and-chatbot
## 🚀 Project Overview
This project is an intelligent web application designed to help users navigate the stock market by providing **predictive analytics** and **sentiment insights** from news.  

Built with **Streamlit**, the application integrates multiple components:  
- **ensemble forecasting model**  
- A **real-time news sentiment analyzer**  
- An interactive **chatbot powered by Google Gemini**  

The goal is to provide a comprehensive, all-in-one tool for both **technical** and **fundamental analysis**.

---

## ✨ Key Features

### 📈 Stock Price Forecasting
- **Ensemble Model**: Combines the strengths of two models:
  - **Prophet** – excels at capturing long-term trends & seasonality.  
  - **XGBoost** – handles short-term fluctuations & non-linear patterns.  
- **Visualizations**: Interactive charts show predicted price movements for user-specified future days.

---

### 📰 News Sentiment Analysis
- **News Aggregation**: Fetches recent stock-related news via the **Finnhub API**.  
- **Sentiment Classification**: A pre-trained **Hugging Face Transformers model** classifies sentiment (Positive / Negative / Neutral).  
- **Insights**: Displays sentiment distribution for quick understanding of market mood.

---

### 🤖 Gemini-Powered Chatbot
- **Conversational Interface**: An intelligent chatbot powered by **Google Gemini API**.  
- **Contextual Answers**: Responds to natural language queries about:
  - Historical data  
  - Forecast predictions  
  - News sentiment  

---

### 📊 Historical Data & Metrics
- **Data Visualization**: View historical stock data using **yfinance** + interactive **Plotly** charts.  
- **Performance Metrics**: Evaluate forecasting models with:
  - RMSE (Root Mean Squared Error)  
  - MAE (Mean Absolute Error)  
  - R² (Coefficient of Determination)  

---

## 🛠 Technology Stack
- **Web Framework**: Streamlit  
- **Data Sources**: yfinance, Finnhub API  
- **Machine Learning**: Prophet, XGBoost, Hugging Face Transformers  
- **APIs**: Google Gemini API  
- **Data Handling**: Pandas  
- **Visualization**: Plotly  

---

## ⚙️ Getting Started

### ✅ Prerequisites
- Python 3.8 or higher  
- Pip package manager  

Install dependencies:
   pip install -r requirements.txt
   
   Set up your API keys in a .env file:
   GEMINI_API_KEY="your_gemini_api_key"
   FINNHUB_API_KEY="your_finnhub_api_key"

▶️ Running the Application
   Start the Streamlit app:
       streamlit run app.py
    
   The application will open in your default web browser. 🎉
