import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_log_error
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = '/Users/paullouw/supply-chain-forecast'

# ── Evaluation metric ─────────────────────────────────────────────────────────
def rmsle(y_true, y_pred):
    """
    Root Mean Squared Logarithmic Error — the official Favorita competition metric.
    Predictions are clipped at 0 to avoid log of negative numbers.
    """
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(model_df):
    """
    Encode categorical features and return feature matrix X, target y,
    and the list of feature column names.

    Label encodes: family, city, state, type.
    store_nbr and cluster are already numeric.
    """
    df = model_df.copy()

    cat_cols = ['family', 'city', 'state', 'type']
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

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

    return X, y, feature_cols, encoders


# ── Time series cross validation ──────────────────────────────────────────────
def ts_cross_validate(model_fn, X, y, n_splits=5):
    """
    Evaluate a model using TimeSeriesSplit cross validation.
    Returns list of RMSLE scores per fold and the mean.

    Uses TimeSeriesSplit to respect temporal ordering — no data leakage.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = model_fn()
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        score = rmsle(y_val, preds)
        scores.append(score)
        print(f"  Fold {fold+1}: RMSLE = {score:.4f}")

    mean_score = np.mean(scores)
    print(f"  Mean RMSLE: {mean_score:.4f}\n")
    return scores, mean_score


# ── Model 1: Ridge Regression ─────────────────────────────────────────────────
def train_ridge(X, y):
    """
    Train Ridge Regression baseline with TimeSeriesSplit CV.
    Target is log1p(sales) — predictions are exponentiated back to raw sales.
    Ridge is trained on log target to handle the right-skewed sales distribution.
    """
    print("=" * 50)
    print("Ridge Regression")
    print("=" * 50)

    y_log = np.log1p(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    def model_fn():
        class RidgeWrapper:
            def __init__(self):
                self.model = Ridge(alpha=1.0)
                self.scaler = StandardScaler()

            def fit(self, X, y):
                self.model.fit(X, np.log1p(y))

            def predict(self, X):
                return np.expm1(self.model.predict(X))

        return RidgeWrapper()

    scores, mean_score = ts_cross_validate(model_fn, X_scaled, y)

    # Final model trained on all data
    final_model = Ridge(alpha=1.0)
    final_model.fit(X_scaled, y_log)

    return final_model, scaler, scores, mean_score


# ── Model 2: XGBoost ──────────────────────────────────────────────────────────
def train_xgboost(X, y):
    """
    Train XGBoost with TimeSeriesSplit CV.
    Uses log1p target transformation consistent with RMSLE objective.
    XGBoost handles feature interactions automatically — no scaling needed.
    """
    print("=" * 50)
    print("XGBoost")
    print("=" * 50)

    def model_fn():
        class XGBWrapper:
            def __init__(self):
                self.model = xgb.XGBRegressor(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0
                )

            def fit(self, X, y):
                self.model.fit(X, np.log1p(y))

            def predict(self, X):
                return np.expm1(self.model.predict(X))

        return XGBWrapper()

    scores, mean_score = ts_cross_validate(model_fn, X, y)

    # Final model
    final_model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    final_model.fit(X, np.log1p(y))

    return final_model, scores, mean_score


# ── Model 3: PyTorch MLP ──────────────────────────────────────────────────────
# class SalesMLP(nn.Module):
#     """
#     Feedforward MLP for tabular sales forecasting.
#     Architecture: input → 256 → 128 → 64 → 1
#     BatchNorm and Dropout used for regularisation.
#     """
#     def __init__(self, input_dim):
#         super(SalesMLP, self).__init__()
#         self.network = nn.Sequential(
#             nn.Linear(input_dim, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(0.3),

#             nn.Linear(256, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Dropout(0.2),

#             nn.Linear(128, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),

#             nn.Linear(64, 1)
#         )

#     def forward(self, x):
#         return self.network(x).squeeze(1)
class SalesMLP(nn.Module):
    """
    Feedforward MLP for tabular sales forecasting.
    Architecture: input → 256 → 128 → 64 → 1
    LayerNorm used instead of BatchNorm1d for MPS compatibility.
    Dropout used for regularisation.
    """
    def __init__(self, input_dim):
        super(SalesMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


def train_mlp(X, y, epochs=20, batch_size=2048, lr=1e-3):
    """
    Train PyTorch MLP with a simple train/val temporal split (80/20).
    Uses MSE loss on log1p target, consistent with RMSLE evaluation.
    Runs on MPS (Apple Silicon) if available, else CPU.
    """
    print("=" * 50)
    print("PyTorch MLP")
    print("=" * 50)

    # Device — use Apple Silicon MPS if available /Users/paullouw/supply-chain-forecast/.venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"
    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device('cpu')
    print(f"  Device: {device}") # hangs
    # device = torch.device('cpu')
    # print("  Note: Using CPU (MPS disabled for stability)")

    y_log = np.log1p(y).astype(np.float32)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    # Temporal train/val split (80/20)
    split = int(len(X_scaled) * 0.8)
    X_train, X_val = X_scaled[:split], X_scaled[split:]
    y_train, y_val = y_log[:split], y_log[split:]

    # DataLoaders
    train_ds = TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_train)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val),
        torch.tensor(y_val)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Model, loss, optimiser
    model = SalesMLP(input_dim=X_scaled.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0
        val_preds = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                val_loss += criterion(preds, y_batch).item()
                val_preds.extend(preds.cpu().numpy())

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        # RMSLE on val set
        val_rmsle = rmsle(np.expm1(y_val), np.expm1(np.array(val_preds)))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs} — Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val RMSLE: {val_rmsle:.4f}")

    # Restore best model
    model.load_state_dict(best_state)
    print(f"  Best Val RMSLE: {rmsle(np.expm1(y_val), np.expm1(np.array(val_preds))):.4f}\n")

    return model, scaler, val_rmsle


# ── Main: train all models and compare ───────────────────────────────────────
# if __name__ == '__main__':
#     print("Loading features...")
#     model_df = pd.read_parquet(f'{BASE_DIR}/data/model_features.parquet')
#     print(f"Shape: {model_df.shape}")

#     X, y, feature_cols, encoders = preprocess(model_df)
#     print(f"Feature matrix: {X.shape}\n")

#     # Train all three models
#     ridge_model, ridge_scaler, ridge_scores, ridge_mean = train_ridge(X, y)
#     xgb_model, xgb_scores, xgb_mean = train_xgboost(X, y)
#     mlp_model, mlp_scaler, mlp_rmsle = train_mlp(X, y)

#     # Summary
#     print("\n" + "=" * 50)
#     print("MODEL COMPARISON SUMMARY")
#     print("=" * 50)
#     print(f"Ridge Regression  — Mean RMSLE: {ridge_mean:.4f}")
#     print(f"XGBoost           — Mean RMSLE: {xgb_mean:.4f}")
#     print(f"PyTorch MLP       — Val  RMSLE: {mlp_rmsle:.4f}")
if __name__ == '__main__':
    print("Loading features...")
    model_df = pd.read_parquet(f'{BASE_DIR}/data/model_features.parquet')
    print(f"Shape: {model_df.shape}")

    X, y, feature_cols, encoders = preprocess(model_df)
    print(f"Feature matrix: {X.shape}\n")

    # Train Ridge and XGBoost on full dataset
    ridge_model, ridge_scaler, ridge_scores, ridge_mean = train_ridge(X, y)
    xgb_model, xgb_scores, xgb_mean = train_xgboost(X, y)

    # MLP on 50k sample — CPU friendly, sufficient for portfolio demonstration
    # print("Sampling 50k rows for MLP training...")
    # mlp_df = model_df.tail(50000)
    # X_mlp, y_mlp, _, _ = preprocess(mlp_df)
    # mlp_model, mlp_scaler, mlp_rmsle = train_mlp(X_mlp, y_mlp, epochs=10)

    # Summary
    print("\n" + "=" * 50)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 50)
    print(f"Ridge Regression  — Mean RMSLE: {ridge_mean:.4f}")
    print(f"XGBoost           — Mean RMSLE: {xgb_mean:.4f}")
    # print(f"PyTorch MLP       — Val  RMSLE: {mlp_rmsle:.4f}")