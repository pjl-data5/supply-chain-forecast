import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = '/Users/paullouw/supply-chain-forecast'

# ── Helpers ───────────────────────────────────────────────────────────────────
def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))


def preprocess(model_df):
    df = model_df.copy()
    cat_cols = ['family', 'city', 'state', 'type']
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    feature_cols = [
        'store_nbr', 'family', 'city', 'state', 'type', 'cluster',
        'onpromotion', 'oil_price', 'is_national_holiday', 'is_payday',
        'is_earthquake_period', 'day_of_week', 'month', 'year',
        'day_of_month', 'week_of_year', 'is_weekend',
        'sales_lag_7', 'sales_lag_14', 'sales_lag_28',
        'sales_roll_mean_7', 'sales_roll_mean_28',
        'family_sales_share', 'family_sales_share_roll_28',
        'family_sales_per_transaction', 'family_sales_per_transaction_roll_28',
        'take_rate_imputed'
    ]

    X = df[feature_cols].values
    y = df['sales'].values
    return X, y, feature_cols


# ── 1. Feature importance ─────────────────────────────────────────────────────
def plot_feature_importance(model, feature_cols):
    importance = model.get_booster().get_score(importance_type='gain')
    importance_named = {feature_cols[int(k[1:])]: v for k, v in importance.items()}
    importance_df = pd.DataFrame({
        'feature': list(importance_named.keys()),
        'importance': list(importance_named.values())
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance_df, x='importance', y='feature')
    plt.title('XGBoost Feature Importance (Gain)')
    plt.xlabel('Gain')
    plt.tight_layout()
    plt.savefig(f'{BASE_DIR}/notebooks/feature_importance.png')
    plt.close()
    print("Saved feature_importance.png")
    return importance_df


# ── 2. Residual analysis ──────────────────────────────────────────────────────
def plot_residuals(y_true, y_pred, dates):
    """
    Plot residuals over time and distribution.
    Helps identify systematic bias in predictions.
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Residuals over time
    axes[0].scatter(dates, residuals, alpha=0.1, s=1)
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_title('Residuals Over Time')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Residual (Actual - Predicted)')

    # Residual distribution
    axes[1].hist(residuals, bins=100, edgecolor='none')
    axes[1].axvline(0, color='red', linestyle='--')
    axes[1].set_title('Residual Distribution')
    axes[1].set_xlabel('Residual')

    plt.tight_layout()
    plt.savefig(f'{BASE_DIR}/notebooks/residuals.png')
    plt.close()
    print("Saved residuals.png")


# ── 3. Actual vs predicted ────────────────────────────────────────────────────
def plot_actual_vs_predicted(y_true, y_pred, dates, family, store_nbr):
    """Plot actual vs predicted sales for a specific store/family combo."""
    plt.figure(figsize=(15, 5))
    plt.plot(dates, y_true, label='Actual', alpha=0.7)
    plt.plot(dates, y_pred, label='Predicted', alpha=0.7)
    plt.title(f'Actual vs Predicted — Store {store_nbr} | {family}')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{BASE_DIR}/notebooks/actual_vs_predicted.png')
    plt.close()
    print("Saved actual_vs_predicted.png")


# ── 4. Take rate analysis ─────────────────────────────────────────────────────
def take_rate_analysis(df):
    """
    Identify chronic understock patterns using take rate.
    Surfaces high take rate + low sales volume SKUs as restock candidates.
    """
    # Aggregate take rate and sales per store/family
    summary = df.groupby(['store_nbr', 'family']).agg(
        avg_sales=('sales', 'mean'),
        avg_take_rate=('family_sales_per_transaction', 'mean'),
        avg_sales_share=('family_sales_share', 'mean'),
        total_sales=('sales', 'sum'),
        imputed_pct=('take_rate_imputed', 'mean')
    ).reset_index()

    # Quartile segmentation
    summary['sales_quartile'] = pd.qcut(summary['avg_sales'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    summary['take_rate_quartile'] = pd.qcut(summary['avg_take_rate'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

    # Flag restock candidates: high take rate (Q3/Q4) + low sales (Q1/Q2)
    summary['restock_candidate'] = (
        (summary['take_rate_quartile'].isin(['Q3', 'Q4'])) &
        (summary['sales_quartile'].isin(['Q1', 'Q2']))
    )

    restock = summary[summary['restock_candidate']].sort_values('avg_take_rate', ascending=False)

    print(f"\nRestock candidates (high take rate, low sales): {len(restock)}")
    print(restock[['store_nbr', 'family', 'avg_sales', 'avg_take_rate', 'avg_sales_share']].head(15).to_string())

    # Plot take rate vs sales by family
    plt.figure(figsize=(12, 8))
    family_summary = df.groupby('family').agg(
        avg_sales=('sales', 'mean'),
        avg_take_rate=('family_sales_per_transaction', 'mean')
    ).reset_index()

    sns.scatterplot(
        data=family_summary,
        x='avg_sales', y='avg_take_rate',
        hue='family', s=200
    )
    for _, row in family_summary.iterrows():
        plt.annotate(row['family'], (row['avg_sales'], row['avg_take_rate']),
                    fontsize=7, ha='right')

    plt.title('Take Rate vs Average Sales by Family')
    plt.xlabel('Average Daily Sales')
    plt.ylabel('Average Take Rate (sales/transaction)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    plt.tight_layout()
    plt.savefig(f'{BASE_DIR}/notebooks/take_rate_analysis.png')
    plt.close()
    print("Saved take_rate_analysis.png")

    return summary, restock


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Loading data...")
    model_df = pd.read_parquet(f'{BASE_DIR}/data/model_features.parquet')
    full_df = pd.read_parquet(f'{BASE_DIR}/data/features.parquet')

    X, y, feature_cols = preprocess(model_df)

    # Train final XGBoost on full data
    print("Training final XGBoost model...")
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X, np.log1p(y))
    preds = np.expm1(model.predict(X))

    # RMSLE on full training data
    print(f"Training RMSLE: {rmsle(y, preds):.4f}")

    # Feature importance
    print("\nGenerating feature importance...")
    importance_df = plot_feature_importance(model, feature_cols)
    print(importance_df.head(10).to_string())

    # Residuals
    print("\nGenerating residual plots...")
    plot_residuals(y, preds, model_df['date'])

    # Actual vs predicted for BEVERAGES store 1
    print("\nActual vs predicted plot...")
    mask = (model_df['family'] == 'BEVERAGES') & (model_df['store_nbr'] == 1)
    plot_actual_vs_predicted(
        y[mask], preds[mask],
        model_df[mask]['date'],
        'BEVERAGES', 1
    )

    # Take rate analysis
    print("\nRunning take rate analysis...")
    summary, restock = take_rate_analysis(full_df)

    print("\nAll evaluation plots saved to notebooks/")