import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.impute import SimpleImputer
import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score, cross_validate
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from data_collection.vader import analyze_sentiment
from utils.logger import Logger

import xgboost as xgb

logger = Logger.setup(__name__)


feature_columns = [
    'MACD', 'Signal', 'RSI', 'ATR', 'OBV', 'Volume',
    'MACD_lag_1', 'MACD_lag_2', 'MACD_lag_3', 'MACD_lag_4', 'MACD_lag_5',
    'Signal_lag_1', 'Signal_lag_2', 'Signal_lag_3', 'Signal_lag_4', 'Signal_lag_5',
    'RSI_lag_1', 'RSI_lag_2', 'RSI_lag_3', 'RSI_lag_4', 'RSI_lag_5',
    'ATR_lag_1', 'ATR_lag_2', 'ATR_lag_3', 'ATR_lag_4', 'ATR_lag_5',
    'OBV_lag_1', 'OBV_lag_2', 'OBV_lag_3', 'OBV_lag_4', 'OBV_lag_5',
    'Volume_lag_1', 'Volume_lag_2', 'Volume_lag_3', 'Volume_lag_4', 'Volume_lag_5',
    'average_sentiment', 'mention_count', 'total_score', 'sentiment_moving_average_3d', 'sentiment_mentions_interaction'
]


def preprocess_features(X, y):
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=X.columns)  # Ensure X is a DataFrame
    
    # Drop NaNs and align y
    X.dropna(inplace=True)
    y = y[X.index]  # Align y with the new X
    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res


def train_xgboost(X_train, y_train):
    xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    param_grid_xgb = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05],
        'max_depth': [3, 5],
        'subsample': [0.6, 0.8],
        'colsample_bytree': [0.6, 0.8]
    }
    grid_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=TimeSeriesSplit(n_splits=5), scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_xgb.fit(X_train, y_train)
    return grid_xgb.best_estimator_


def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }
    grid_rf = GridSearchCV(rf, param_grid, cv=TimeSeriesSplit(n_splits=5), scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_rf.fit(X_train, y_train)
    return grid_rf.best_estimator_

def evaluate_model(model, X_test, y_test, model_name='Model'):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"--- {model_name} Evaluation ---")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC Score: {auc}\n")

    # Additional Metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}\n")
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')

    os.makedirs('models', exist_ok=True)

    # Save the plot instead of showing it
    plt.savefig(f'models/{model_name}_roc_curve.png')
    plt.close()
    print(f"ROC Curve for {model_name} saved to plots/{model_name}_roc_curve.png\n")

    # Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, label=f'{model_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc='lower left')
    plt.tight_layout()
    
    os.makedirs('plots/precision_recall_curves', exist_ok=True)
    
    pr_plot_path = f'plots/precision_recall_curves/{model_name}_precision_recall_curve.png'
    plt.savefig(pr_plot_path)
    plt.close()
    print(f"Precision-Recall Curve for {model_name} saved to {pr_plot_path}\n")
    #plt.show()

def load_data(filepaths):
    logger.info("Loading merged data from filepaths...")
    dataframes = []
    for filepath in filepaths:
        try:
            df = pd.read_csv(filepath, parse_dates=['Date'])
            dataframes.append(df)
            print(f"Loaded {filepath} with shape {df.shape}")
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    combined_data = pd.concat(dataframes, ignore_index=True)
    print(f"Combined data shape: {combined_data.shape}")

    # Ensure consistent datetime format
    combined_data['Date'] = pd.to_datetime(combined_data['Date'], utc=True)
    
    # Inspect Column Names
    print("Combined Data Columns:", combined_data.columns.tolist())
  
    return combined_data

def perform_cross_validation(model, X, y):
    """
    Perform cross-validation and return evaluation metrics.
    
    Args:
        model: Machine learning model to evaluate.
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.
    
    Returns:
        dict: Cross-validation scores for different metrics.
    """
    tscv = TimeSeriesSplit(n_splits=5)
    scoring = ['roc_auc', 'precision', 'recall', 'f1']
    
    cv_results = cross_validate(model, X, y, cv=tscv, scoring=scoring, n_jobs=-1, verbose=1)
    
    print("Cross-Validation Results:")
    for metric in scoring:
        scores = cv_results[f'test_{metric}']
        print(f"{metric.capitalize()}: {scores.mean():.4f} ± {scores.std():.4f}")
    
    return cv_results


def simulate_paper_trading(model, X_test, y_test, initial_capital=1000):
    """
    Simulate paper trading based on model predictions.
    
    Strategy:
    - When the model predicts a positive signal (1), buy a $100 position.
    - Hold the position for one day.
    - Calculate daily returns based on actual price movement.
    
    Args:
        model: Trained machine learning model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Actual target values.
        initial_capital (float): Starting capital for simulation.
    """
    print(f"\n--- Paper Trading Simulation for {model.__class__.__name__} ---")
    
    # Assuming 'Date' and 'Close' price are available in the data
    # Modify as per your actual data columns
    # Here, we'll assume that 'Close' price is available
    # If not, you need to include it during data loading
    if 'Close' not in X_test.columns:
        print("Error: 'Close' price column not found in test features. Cannot perform paper trading simulation.")
        return
    
    # Align X_test and y_test
    X_test = X_test.copy()
    X_test['Target'] = y_test
    X_test.reset_index(drop=True, inplace=True)
    
    # Predict probabilities and classes
    predictions = model.predict(X_test)
    predicted_probs = model.predict_proba(X_test)[:, 1]
    
    # Initialize capital and positions
    capital = initial_capital
    position_size = 100  # $100 per trade
    portfolio = []
    trades = 0
    
    for i in range(len(X_test) - 1):
        if predictions[i] == 1:
            # Buy at closing price of day i
            buy_price = X_test.loc[i, 'Close']
            # Sell at closing price of day i+1
            sell_price = X_test.loc[i + 1, 'Close']
            profit = sell_price - buy_price
            capital += profit
            trades += 1
            portfolio.append({
                'Buy_Date': X_test.loc[i, 'Date'],
                'Sell_Date': X_test.loc[i + 1, 'Date'],
                'Buy_Price': buy_price,
                'Sell_Price': sell_price,
                'Profit': profit
            })
    
    # Convert portfolio to DataFrame
    portfolio_df = pd.DataFrame(portfolio)
    
    # Calculate total profit/loss
    total_profit = portfolio_df['Profit'].sum()
    print(f"Total Trades Executed: {trades}")
    print(f"Total Profit/Loss: ${total_profit:.2f}")
    print(f"Final Capital: ${capital:.2f}")
    
    # Plot portfolio performance
    if not portfolio_df.empty:
        portfolio_df['Cumulative Profit'] = portfolio_df['Profit'].cumsum()
        plt.figure(figsize=(10, 6))
        plt.plot(portfolio_df['Sell_Date'], portfolio_df['Cumulative Profit'], marker='o')
        plt.title(f'Cumulative Profit - {model.__class__.__name__}')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Profit ($)')
        plt.tight_layout()
        
        os.makedirs('plots/paper_trading', exist_ok=True)
        plot_path = f"plots/paper_trading/{model.__class__.__name__}_cumulative_profit.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Cumulative Profit plot saved to {plot_path}\n")
    else:
        print("No trades were executed based on the model's predictions.\n")

   

def main():
    ## List of processed data file paths for each ticker
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    filepaths = [f'data/processed/merged/{ticker}_merged.csv' for ticker in tickers]
    #sentiment_filepaths = [f'data/interim/reddit/subreddits23_mentions/{ticker}_mentions.csv' for ticker in tickers]

    data = load_data(filepaths)

    # Features and target
    X = data[feature_columns]
    y = data['Target']
    
    # Train-Test Split
    train_size = int(len(data) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Preprocess features
    X_train_res, y_train_res = preprocess_features(X_train, y_train)

    models = {}    
    # Train and Evaluate Random Forest
    try:
        best_rf = train_random_forest(X_train_res, y_train_res)
        models['Random Forest'] = best_rf

        importances_rf = best_rf.feature_importances_
        feature_names_rf = X_train_res.columns
        feature_importances_rf = pd.Series(importances_rf, index=feature_names_rf).sort_values(ascending=False)
        print("Random Forest Feature Importances:")
        print(feature_importances_rf.head(10))
            
        #print("Best Random Forest Parameters:", best_rf.get_params())

        rf_cv_results = perform_cross_validation(best_rf, X_train_res, y_train_res)
        
        evaluate_model(best_rf, X_test, y_test, model_name='Random Forest')
    except Exception as e:
        print(f"Error training or evaluating Random Forest: {e}")
    
    # Train and Evaluate XGBoost
    try:
        best_xgb = train_xgboost(X_train_res, y_train_res)
        models['XGBoost'] = best_xgb
        
        xgb.plot_importance(best_xgb)
        plt.savefig('plots/XGBoost_feature_importance.png')
        plt.close()
        print("Best XGBoost Parameters:", best_xgb.get_params())

        xgb_cv_results = perform_cross_validation(best_xgb, X_train_res, y_train_res)
        evaluate_model(best_xgb, X_test, y_test, model_name='XGBoost')
    except Exception as e:
        print(f"Error training or evaluating XGBoost: {e}")
    
    # Save Models
    try:
        joblib.dump(best_rf, 'models/random_forest_model.pkl')
        joblib.dump(best_xgb, 'models/xgboost_model.pkl')
        print("Models saved successfully.")
    except Exception as e:
        print(f"Error saving models: {e}")

      # Paper Trading Simulation
    try:
        simulate_paper_trading(best_rf, X_test, y_test, initial_capital=1000)
        simulate_paper_trading(best_xgb, X_test, y_test, initial_capital=1000)
    except Exception as e:
        print(f"Error during paper trading simulation: {e}")

if __name__ == "__main__":
    main()