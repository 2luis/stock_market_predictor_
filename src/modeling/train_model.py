"""
Model training script that implements and evaluates machine learning models for stock price prediction.
This script handles the training of Random Forest and XGBoost models, performs cross-validation,
and simulates paper trading based on model predictions. The models use both technical indicators
and sentiment data as features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.impute import SimpleImputer
import os
import sys

# add the src directory to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score, cross_validate
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_score, recall_score, f1_score, 
                           precision_recall_curve)
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from utils.logger import Logger

import xgboost as xgb

logger = Logger.setup(__name__)

# define feature columns used for model training
feature_columns = [
    'MACD', 'Signal', 'RSI', 'ATR', 'OBV', 'Volume',
    'MACD_lag_1', 'MACD_lag_2', 'MACD_lag_3', 'MACD_lag_4', 'MACD_lag_5',
    'Signal_lag_1', 'Signal_lag_2', 'Signal_lag_3', 'Signal_lag_4', 'Signal_lag_5',
    'RSI_lag_1', 'RSI_lag_2', 'RSI_lag_3', 'RSI_lag_4', 'RSI_lag_5',
    'ATR_lag_1', 'ATR_lag_2', 'ATR_lag_3', 'ATR_lag_4', 'ATR_lag_5',
    'OBV_lag_1', 'OBV_lag_2', 'OBV_lag_3', 'OBV_lag_4', 'OBV_lag_5',
    'Volume_lag_1', 'Volume_lag_2', 'Volume_lag_3', 'Volume_lag_4', 'Volume_lag_5',
    'average_sentiment', 'mention_count', 'total_score', 
    'sentiment_moving_average_3d', 'sentiment_mentions_interaction'
]

def preprocess_features(X, y):
    """
    preprocess features by handling missing values and class imbalance.
    
    args:
        X (pd.DataFrame): feature matrix
        y (pd.Series): target variable
    
    returns:
        tuple: processed features and target (X_res, y_res)
    """
    logger.info("Starting feature preprocessing...")
    
    # handle missing values using mean imputation
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=X.columns)
    
    # drop remaining nans and align target
    X.dropna(inplace=True)
    y = y[X.index]
    
    # handle class imbalance using SMOTE
    logger.info("Applying SMOTE for class balance...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    logger.info("Feature preprocessing completed")
    return X_res, y_res


def train_xgboost(X_train, y_train):
    """
    train xgboost classifier with grid search cross validation.
    
    args:
        X_train (pd.DataFrame): training features
        y_train (pd.Series): training target variable
    
    returns:
        XGBClassifier: best performing xgboost model
    """
    logger.info("Training XGBoost model with grid search...")
    
    # initialize base model
    xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    
    # define parameter grid for search
    param_grid_xgb = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05],
        'max_depth': [3, 5],
        'subsample': [0.6, 0.8],
        'colsample_bytree': [0.6, 0.8]
    }
    
    # perform grid search with time series cross validation
    grid_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=TimeSeriesSplit(n_splits=5), 
                           scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_xgb.fit(X_train, y_train)
    
    logger.info("XGBoost training completed")
    return grid_xgb.best_estimator_


def train_random_forest(X_train, y_train):
    """
    train random forest classifier with grid search cross validation.
    
    args:
        X_train (pd.DataFrame): training features
        y_train (pd.Series): training target variable
    
    returns:
        RandomForestClassifier: best performing random forest model
    """
    logger.info("Training Random Forest model with grid search...")
    
    # initialize base model
    rf = RandomForestClassifier(random_state=42)
    
    # define parameter grid for search
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }
    
    # perform grid search with time series cross validation
    grid_rf = GridSearchCV(rf, param_grid, cv=TimeSeriesSplit(n_splits=5), 
                          scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_rf.fit(X_train, y_train)
    
    logger.info("Random Forest training completed")
    return grid_rf.best_estimator_

def evaluate_model(model, X_test, y_test, model_name='Model'):
    """
    evaluate model performance using various metrics and generate visualization plots.
    
    args:
        model: trained classifier model
        X_test (pd.DataFrame): test features
        y_test (pd.Series): test target variable
        model_name (str): name of the model for plotting
    """
    print(f"--- {model_name} Evaluation ---")
    # generate predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    

    # calculate metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # log performance metrics
    logger.info(f"ROC AUC Score: {auc:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    
    # generate and save roc curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    
    # save roc curve plot
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/{model_name}_roc_curve.png')
    plt.close()
    
    # generate and save precision-recall curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, label=f'{model_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc='lower left')
    
    # save precision-recall curve plot
    os.makedirs('plots/precision_recall_curves', exist_ok=True)
    plt.savefig(f'plots/precision_recall_curves/{model_name}_precision_recall_curve.png')
    plt.close()
    
    logger.info("Model evaluation completed")

def load_data(filepaths):
    """
    load and combine data from multiple csv files.
    
    args:
        filepaths (list): list of file paths to load data from
    
    returns:
        pd.DataFrame: combined dataset with consistent datetime format
    """
    logger.info("Loading merged data from filepaths...")
    
    # load and combine all dataframes
    dataframes = []
    for filepath in filepaths:
        try:
            df = pd.read_csv(filepath, parse_dates=['Date'])
            dataframes.append(df)
            logger.info(f"Loaded {filepath} with shape {df.shape}")
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
    
    # combine all dataframes
    combined_data = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Combined data shape: {combined_data.shape}")

    # ensure consistent datetime format
    combined_data['Date'] = pd.to_datetime(combined_data['Date'], utc=True)
    
    return combined_data

def perform_cross_validation(model, X, y):
    """
    perform time series cross-validation and return evaluation metrics.
    
    args:
        model: machine learning model to evaluate
        X (pd.DataFrame): features
        y (pd.Series): target variable
    
    returns:
        dict: cross-validation scores for different metrics
    """
    logger.info("Starting cross-validation...")
    
    # setup time series cross validation
    tscv = TimeSeriesSplit(n_splits=5)
    scoring = ['roc_auc', 'precision', 'recall', 'f1']
    
    # perform cross validation
    cv_results = cross_validate(model, X, y, cv=tscv, scoring=scoring, n_jobs=-1, verbose=1)
    
    # log results
    for metric in scoring:
        scores = cv_results[f'test_{metric}']
        logger.info(f"{metric.capitalize()}: {scores.mean():.4f} ± {scores.std():.4f}")
    
    logger.info("Cross-validation completed")
    return cv_results

def main():
    """
    main function to orchestrate the model training and evaluation pipeline.
    handles data loading, model training, evaluation, and paper trading simulation.
    """
    # setup data paths
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    filepaths = [f'data/processed/merged/{ticker}_merged.csv' for ticker in tickers]
    
    # load and prepare data
    logger.info("Starting model training pipeline...")
    data = load_data(filepaths)

    # prepare features and target
    X = data[feature_columns]
    y = data['Target']
    
    # perform train-test split
    train_size = int(len(data) * 0.8)
    X_train_res, X_test = X[:train_size], X[train_size:]
    y_train_res, y_test = y[:train_size], y[train_size:]
    
    models = {}    
    
    # train and evaluate random forest
    try:
        logger.info("Training Random Forest model...")
        best_rf = train_random_forest(X_train_res, y_train_res)
        models['Random Forest'] = best_rf

        # analyze feature importance
        importances_rf = best_rf.feature_importances_
        feature_names_rf = X_train_res.columns
        feature_importances_rf = pd.Series(importances_rf, index=feature_names_rf).sort_values(ascending=False)
        logger.info("Random Forest Feature Importances:")
        logger.info(feature_importances_rf.head(10))

        # perform cross validation and evaluation
        rf_cv_results = perform_cross_validation(best_rf, X_train_res, y_train_res)
        evaluate_model(best_rf, X_test, y_test, model_name='Random Forest')
    except Exception as e:
        logger.error(f"Error training or evaluating Random Forest: {e}")
    
    # train and evaluate xgboost
    try:
        logger.info("Training XGBoost model...")
        best_xgb = train_xgboost(X_train_res, y_train_res)
        models['XGBoost'] = best_xgb
        
        # plot feature importance
        xgb.plot_importance(best_xgb)
        plt.savefig('plots/XGBoost_feature_importance.png')
        plt.close()
        logger.info("Best XGBoost Parameters:", best_xgb.get_params())

        # perform cross validation and evaluation
        xgb_cv_results = perform_cross_validation(best_xgb, X_train_res, y_train_res)
        evaluate_model(best_xgb, X_test, y_test, model_name='XGBoost')
    except Exception as e:
        logger.error(f"Error training or evaluating XGBoost: {e}")
    
    # save trained models
    try:
        logger.info("Saving trained models...")
        joblib.dump(best_rf, 'models/random_forest_model.pkl')
        joblib.dump(best_xgb, 'models/xgboost_model.pkl')
        logger.info("Models saved successfully")
    except Exception as e:
        logger.error(f"Error saving models: {e}")

if __name__ == "__main__":
    main()