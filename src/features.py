import pandas as pd
import numpy as np

BASE_DIR = '/Users/paullouw/supply-chain-forecast'
DATA_DIR = f'{BASE_DIR}/data'

def load_data():
    train = pd.read_csv(f'{DATA_DIR}/train.csv', parse_dates=['date'])
    stores = pd.read_csv(f'{DATA_DIR}/stores.csv')
    oil = pd.read_csv(f'{DATA_DIR}/oil.csv', parse_dates=['date'])
    holidays = pd.read_csv(f'{DATA_DIR}/holidays_events.csv', parse_dates=['date'])
    transactions = pd.read_csv(f'{DATA_DIR}/transactions.csv', parse_dates=['date'])
    return train, stores, oil, holidays, transactions


# Clean oil prices (forward fill missing weekends/holidays)
def clean_oil(oil):
    oil = oil.set_index('date').resample('D').mean().ffill().reset_index()
    oil.columns = ['date', 'oil_price']
    return oil

def clean_holidays(holidays):
    # Drop transferred holidays - they're essentially normal days
    holidays = holidays[holidays['transferred'] == False].copy()
    
    # Keep only national holidays for simplicity (biggest signal)
    national = holidays[holidays['locale'] == 'National'][['date', 'type']].copy()
    national['is_national_holiday'] = 1
    national = national[['date', 'is_national_holiday']].drop_duplicates()
    return national


# Core feature engineering 
def build_features(train, stores, oil, holidays, transactions, top_n_families=10):
    """
    Build the full feature matrix for demand forecasting and take rate analysis.

    Filters to the top N product families by total sales volume, then engineers
    a rich set of features by merging supplementary datasets and deriving temporal,
    promotional, and behavioural signals. The log-transformed target (sales_log)
    aligns with the RMSLE evaluation metric used in the Favorita competition.

    Parameters
    ----------
    train : pd.DataFrame
        Raw training data with columns: date, store_nbr, family, sales, onpromotion.
    stores : pd.DataFrame
        Store metadata with columns: store_nbr, city, state, type, cluster.
    oil : pd.DataFrame
        Daily oil prices with columns: date, dcoilwtico. Missing values are forward-filled.
    holidays : pd.DataFrame
        Ecuadorian holidays and events. Transferred holidays are excluded as they
        represent normal trading days.
    transactions : pd.DataFrame
        Daily transaction counts per store with columns: date, store_nbr, transactions.
        Used to compute take rate proxies at store and family level. Missing transaction
        values are imputed using a 28-day rolling median per store/family group.
    top_n_families : int, optional
        Number of top product families to include, ranked by total sales. Default is 10.

    Returns
    -------
    df : pd.DataFrame
        Full feature matrix including take rate features and all engineered columns.
        Use this for the Take Rate Analysis dashboard tab.
    model_df : pd.DataFrame
        Model-ready subset with NaNs dropped on all training features and target.
        Use this for model training and evaluation.
    top_families : list of str
        The product family names included in the filtered feature matrix.

    Notes
    -----
    - Lag and rolling features will produce NaN values for early dates in each
      store/family group. These are dropped in model_df but retained in df.
    - The earthquake period (2016-04-16 to 2016-05-16) is flagged as a binary
      feature to help models account for the demand shock from the April 2016
      magnitude 7.8 earthquake in Ecuador.
    - Two take rate proxies are computed since transactions is only available at
      store level, not family level:
        * family_sales_per_transaction: family sales divided by total store transactions.
          Measures how many units of this family were sold per customer visit.
        * family_sales_share: family sales as a share of total store sales on that day.
          Measures the relative contribution of this family to overall store revenue.
      Both metrics are also computed as 28-day rolling averages to capture trends.
    - Missing transaction values (~8% of data, typically holidays/closures) are imputed
      using a 28-day rolling median per store/family. A binary flag take_rate_imputed
      is added so models can discount imputed values during training.
    """

    # Filter to top N families by total sales
    top_families = (
        train.groupby('family')['sales']
        .sum()
        .sort_values(ascending=False)
        .head(top_n_families)
        .index.tolist()
    )
    print(f"Top {top_n_families} families: {top_families}")
    df = train[train['family'].isin(top_families)].copy()

    # ── Merge supplementary data ──
    df = df.merge(stores, on='store_nbr', how='left')
    df = df.merge(clean_oil(oil), on='date', how='left')
    df = df.merge(clean_holidays(holidays), on='date', how='left')
    df = df.merge(transactions, on=['date', 'store_nbr'], how='left')
    df['is_national_holiday'] = df['is_national_holiday'].fillna(0).astype(int)

    # ── Date features ──
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_month'] = df['date'].dt.day
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # ── Payday flag (15th and last day of month) ──
    df['is_payday'] = ((df['day_of_month'] == 15) |
                       (df['day_of_month'] == df['date'].dt.days_in_month)).astype(int)

    # ── Earthquake flag ──
    df['is_earthquake_period'] = (
        (df['date'] >= '2016-04-16') &
        (df['date'] <= '2016-05-16')
    ).astype(int)

    # ── Lag features (per store/family) ──
    df = df.sort_values(['store_nbr', 'family', 'date'])
    for lag in [7, 14, 28]:
        df[f'sales_lag_{lag}'] = (
            df.groupby(['store_nbr', 'family'])['sales']
            .shift(lag)
        )

    # ── Rolling mean features (per store/family) ──
    for window in [7, 28]:
        df[f'sales_roll_mean_{window}'] = (
            df.groupby(['store_nbr', 'family'])['sales']
            .shift(1)
            .groupby([df['store_nbr'], df['family']])
            .transform(lambda x: x.rolling(window).mean())
        )

    # ── Take rate proxy 1: family sales per store transaction ──
    # Flag rows where transactions were missing before imputation
    df['take_rate_imputed'] = df['transactions'].isna().astype(int)

    # Impute missing transactions using 28-day rolling median per store/family
    # Rationale: missing transactions are typically holidays/closures where
    # sales are also low — rolling median captures typical store behaviour
    df['transactions'] = (
        df.groupby(['store_nbr', 'family'])['transactions']
        .transform(lambda x: x.fillna(x.rolling(28, min_periods=1).median()))
    )

    df['family_sales_per_transaction'] = df['sales'] / (df['transactions'] + 1)

    df['family_sales_per_transaction_roll_28'] = (
        df['family_sales_per_transaction']
        .shift(1)
        .groupby([df['store_nbr'], df['family']])
        .transform(lambda x: x.rolling(28).mean())
    )

    # ── Take rate proxy 2: family share of total store sales ──
    store_daily_sales = df.groupby(['date', 'store_nbr'])['sales'].transform('sum')
    df['family_sales_share'] = df['sales'] / (store_daily_sales + 1)

    df['family_sales_share_roll_28'] = (
        df['family_sales_share']
        .shift(1)
        .groupby([df['store_nbr'], df['family']])
        .transform(lambda x: x.rolling(28).mean())
    )

    # ── Log transform target (aligns with RMSLE evaluation) ──
    df['sales_log'] = np.log1p(df['sales'])

    # ── Define model features ──
    model_features = [
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

    model_df = df[model_features + ['date', 'sales', 'sales_log']].dropna()

    print(f"\nModel-ready shape (after dropping NaNs): {model_df.shape}")
    print(f"Date range: {model_df['date'].min()} to {model_df['date'].max()}")
    print(f"Imputed take rate rows: {df['take_rate_imputed'].sum()} ({df['take_rate_imputed'].mean()*100:.1f}%)")

    return df, model_df, top_families


if __name__ == '__main__':
    train, stores, oil, holidays, transactions = load_data()
    df, model_df, top_families = build_features(train, stores, oil, holidays, transactions)
    df.to_parquet(f'{BASE_DIR}/data/features.parquet', index=False)
    model_df.to_parquet(f'{BASE_DIR}/data/model_features.parquet', index=False)
    print("\nSaved features.parquet and model_features.parquet")