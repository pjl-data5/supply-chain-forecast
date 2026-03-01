# supply-chain-forecast
Weekend project. Predicting take rates

# Supply Chain Demand Forecasting Dashboard

End-to-end demand forecasting and take rate analysis built on the Corporación Favorita dataset.

## Stack
Python | Scikit-learn | XGBoost | PyTorch | Streamlit

## Project Structure
- `src/` - Feature engineering, models, evaluation
- `notebooks/` - EDA
- `app.py` - Streamlit dashboard

*Work in progress*

overstock vs understock — and that's exactly where take rate becomes a powerful lens.
In this dataset, take rate isn't explicitly given, but we can engineer it from what we have:

transactions = number of customer visits (proxy for demand opportunity)
sales = actual units sold

take_rate = sales / transactions


This tells you: given how many people walked into the store, how much of a product family actually got bought? A dropping take rate on a high-demand family suggests a stock availability problem, not a demand problem. That's a critical distinction.

Project frame:
Two things on top of the base forecast:
1. Demand Forecast — predict raw sales using Ridge/XGBoost with lag features, promotions, oil price, holidays. Evaluated with RMSLE as Kaggle suggests.
2. Take Rate Analysis Layer — engineer take rate per store/family, identify chronic understock patterns, surface "high take rate + low sales volume" SKUs as restock candidates.
The Streamlit dashboard then has two tabs — Forecast and Take Rate Insights — which makes it genuinely interesting for LinkedIn and directly transferable to your day job narrative.