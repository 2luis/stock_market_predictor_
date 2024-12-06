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
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
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
    
    # Train and Evaluate Random Forest
    try:
        best_rf = train_random_forest(X_train_res, y_train_res)
        importances_rf = best_rf.feature_importances_
        feature_names_rf = X_train_res.columns
        feature_importances_rf = pd.Series(importances_rf, index=feature_names_rf).sort_values(ascending=False)
        print("Random Forest Feature Importances:")
        print(feature_importances_rf.head(10))
            
        #print("Best Random Forest Parameters:", best_rf.get_params())
        
        evaluate_model(best_rf, X_test, y_test, model_name='Random Forest')
    except Exception as e:
        print(f"Error training or evaluating Random Forest: {e}")
    
    # Train and Evaluate XGBoost
    try:
        best_xgb = train_xgboost(X_train_res, y_train_res)
        
        xgb.plot_importance(best_xgb)
        plt.savefig('plots/XGBoost_feature_importance.png')
        plt.close()
        print("Best XGBoost Parameters:", best_xgb.get_params())
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

if __name__ == "__main__":
    main()