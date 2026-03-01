import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = '/Users/paullouw/supply-chain-forecast'
DATA_DIR = os.path.join(BASE_DIR, 'data')

train = pd.read_csv(f'{DATA_DIR}/train.csv', parse_dates=['date'])
stores = pd.read_csv(f'{DATA_DIR}/stores.csv')
oil = pd.read_csv(f'{DATA_DIR}/oil.csv', parse_dates=['date'])
holidays = pd.read_csv(f'{DATA_DIR}/holidays_events.csv', parse_dates=['date'])
transactions = pd.read_csv(f'{DATA_DIR}/transactions.csv', parse_dates=['date'])
# Basic overview
print("=== TRAIN ===")
print(train.shape)
print(train.head())
print(train.dtypes)
print(train.isnull().sum())

print("\n=== STORES ===")
print(stores.head())
print(stores['type'].value_counts())

print("\n=== HOLIDAYS ===")
print(holidays['type'].value_counts())
print(holidays['locale'].value_counts())

print("\n=== OIL ===")
print(oil.isnull().sum())  # oil has missing values, we'll need to handle this

print("\n=== DATE RANGE ===")
print(f"Train: {train['date'].min()} to {train['date'].max()}")
print(f"Unique stores: {train['store_nbr'].nunique()}")
print(f"Unique product families: {train['family'].nunique()}")
print(f"Unique dates: {train['date'].nunique()}")


# 1. Total sales over time
daily_sales = train.groupby('date')['sales'].sum().reset_index()

plt.figure(figsize=(15, 4))
plt.plot(daily_sales['date'], daily_sales['sales'])
plt.title('Total Daily Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.tight_layout()
plt.savefig('notebooks/total_daily_sales.png')
plt.close()

# 2. Sales by product family
family_sales = train.groupby('family')['sales'].sum().sort_values(ascending=False)

plt.figure(figsize=(15, 5))
sns.barplot(x=family_sales.values, y=family_sales.index)
plt.title('Total Sales by Product Family')
plt.xlabel('Total Sales')
plt.tight_layout()
plt.savefig('notebooks/sales_by_family.png')
plt.close()

# 3. Sales by store type
train_stores = train.merge(stores, on='store_nbr')
type_sales = train_stores.groupby('type')['sales'].sum().sort_values(ascending=False)

plt.figure(figsize=(8, 4))
sns.barplot(x=type_sales.index, y=type_sales.values)
plt.title('Total Sales by Store Type')
plt.tight_layout()
plt.savefig('notebooks/sales_by_store_type.png')
plt.close()

# 4. Oil price over time
oil_filled = oil.set_index('date').resample('D').mean().ffill()

plt.figure(figsize=(15, 4))
plt.plot(oil_filled.index, oil_filled['dcoilwtico'])
plt.title('Oil Price Over Time')
plt.axvline(pd.Timestamp('2016-04-16'), color='red', linestyle='--', label='Earthquake')
plt.legend()
plt.tight_layout()
plt.savefig('notebooks/oil_price.png')
plt.close()

# 5. Sales distribution (log scale)
plt.figure(figsize=(10, 4))
train[train['sales'] > 0]['sales'].apply('log1p').hist(bins=50)
plt.title('Log Sales Distribution (excluding zeros)')
plt.xlabel('log1p(sales)')
plt.tight_layout()
plt.savefig('notebooks/sales_distribution.png')
plt.close()

print("Plots saved to notebooks/")

# 6. Zero sales analysis
zero_pct = (train['sales'] == 0).sum() / len(train) * 100
print(f"\nZero sales rows: {zero_pct:.1f}%")

# 7. Promotion impact
print("\nAvg sales - promoted vs not:")
print(train.groupby(train['onpromotion'] > 0)['sales'].mean())