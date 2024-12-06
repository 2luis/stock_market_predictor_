import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import pointbiserialr

# Set aesthetic parameters for plots
sns.set(style="whitegrid", palette="muted", color_codes=True)
plt.rcParams['figure.figsize'] = (12, 8)

def load_data(filepath):
    """
    Load the merged CSV data into a pandas DataFrame.
    """
    try:
        data = pd.read_csv(filepath, parse_dates=['Date'])
        print(f"Data loaded successfully from {filepath}. Shape: {data.shape}")
        return data
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def check_missing_values(data):
    """
    Check for missing values in the DataFrame and visualize them.
    """
    missing = data.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print("Missing Values:\n", missing)
    
    if not missing.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=missing.values, y=missing.index, palette='viridis')
        plt.title('Missing Values by Feature')
        plt.xlabel('Number of Missing Values')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig('plots/missing_values.png')
        plt.close()
        print("Missing values plot saved to plots/missing_values.png")
    else:
        print("No missing values found.")

def plot_feature_distributions(data, feature_columns):
    """
    Plot the distribution of each feature in the DataFrame.
    """
    num_features = len(feature_columns)
    num_cols = 5
    num_rows = num_features // num_cols + int(num_features % num_cols > 0)
    plt.figure(figsize=(20, num_rows * 4))
    
    for idx, feature in enumerate(feature_columns):
        plt.subplot(num_rows, num_cols, idx + 1)
        sns.histplot(data[feature], kde=True, bins=30, color='skyblue')
        plt.title(f'Distribution of {feature}')
    
    plt.tight_layout()
    plt.savefig('plots/feature_distributions.png')
    plt.close()
    print("Feature distributions plot saved to plots/feature_distributions.png")

def plot_correlation_matrix(data, feature_columns):
    """
    Plot the correlation matrix heatmap for the specified features with annotations.
    """
    plt.figure(figsize=(20, 15))
    corr_matrix = data[feature_columns].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('plots/correlation_matrix.png')
    plt.close()
    print("Correlation matrix heatmap saved to plots/correlation_matrix.png")

def plot_target_distribution(data):
    """
    Plot the distribution of the target variable.
    """
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Target', data=data, palette='viridis')
    plt.title('Distribution of Target Variable')
    plt.xlabel('Target')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('plots/target_distribution.png')
    plt.close()
    print("Target distribution plot saved to plots/target_distribution.png")

def plot_features_vs_target(data, feature_columns):
    """
    Plot violin plots of features against the target variable to visualize distribution spread.
    """
    for feature in feature_columns:
        plt.figure(figsize=(10, 6))
        sns.violinplot(x='Target', y=feature, data=data, palette='viridis')
        plt.title(f'{feature} vs Target')
        plt.xlabel('Target')
        plt.ylabel(feature)
        plt.tight_layout()
        plot_filename = f'plots/{feature}_vs_target_violin.png'
        plt.savefig(plot_filename)
        plt.close()
        print(f"Violin plot for {feature} vs Target saved to {plot_filename}")

def plot_categorical_features(data, categorical_features):
    """
    Plot count plots for categorical features against the target variable.
    """
    for feature in categorical_features:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=feature, hue='Target', data=data, palette='viridis')
        plt.title(f'{feature} Distribution by Target')
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.legend(title='Target')
        plt.tight_layout()
        plot_filename = f'plots/{feature}_distribution_by_target.png'
        plt.savefig(plot_filename)
        plt.close()
        print(f"Count plot for {feature} by Target saved to {plot_filename}")

def plot_time_series(data, date_column, features):
    """
    Plot time series for selected features.
    """
    for feature in features:
        plt.figure(figsize=(15, 6))
        sns.lineplot(x=date_column, y=feature, data=data, label=feature)
        plt.title(f'Time Series of {feature}')
        plt.xlabel('Date')
        plt.ylabel(feature)
        plt.legend()
        plt.tight_layout()
        plot_filename = f'plots/time_series_{feature}.png'
        plt.savefig(plot_filename)
        plt.close()
        print(f"Time series plot for {feature} saved to {plot_filename}")

def plot_feature_correlation_with_target(data, feature_columns, target_col='Target'):
    """
    If target is binary (0/1), use point-biserial correlation to measure correlation 
    between each numeric feature and the binary target.
    
    Adjust this method if target is continuous or multiclass.
    """
    # Check if target is binary
    if data[target_col].nunique() == 2:
        correlations = []
        for f in feature_columns:
            # Ensure feature is numeric
            if pd.api.types.is_numeric_dtype(data[f]):
                corr_val, _ = pointbiserialr(data[target_col], data[f])
                correlations.append((f, corr_val))
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        top_features = [x[0] for x in correlations]
        top_corr_values = [x[1] for x in correlations]
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x=top_corr_values, y=top_features, palette='viridis')
        plt.title('Point-Biserial Correlation with Target')
        plt.xlabel('Correlation')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig('plots/feature_correlation_with_target.png')
        plt.close()
        print("Feature correlation with target plot saved to plots/feature_correlation_with_target.png")
    else:
        print("Target is not binary; adjust correlation method as needed.")

def plot_feature_pairplots(data, feature_columns, target_col='Target', top_n=5):
    """
    Create a pairplot of the top_n correlated features with the target (if binary),
    to see pairwise relationships and distribution.
    """
    if data[target_col].nunique() == 2:
        # Compute point-biserial correlation again to identify top features
        correlations = []
        for f in feature_columns:
            if pd.api.types.is_numeric_dtype(data[f]):
                corr_val, _ = pointbiserialr(data[target_col], data[f])
                correlations.append((f, corr_val))
        
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        # Take the top_n features
        top_features = [x[0] for x in correlations[:top_n]]
        
        # Create a pairplot
        sns.pairplot(data, vars=top_features, hue=target_col, palette='viridis')
        plt.savefig('plots/top_features_pairplot.png')
        plt.close()
        print("Top correlated features pairplot saved to plots/top_features_pairplot.png")
    else:
        print("Target is not binary; pairplot might not be meaningful without adjustment.")

def plot_rolling_features(data, date_column, features, window=30):
    """
    Plot rolling mean for selected features to identify trends over time.
    """
    rolling_data = data[[date_column] + features].copy()
    for f in features:
        rolling_data[f'{f}_rolling_mean'] = rolling_data[f].rolling(window=window).mean()

    plt.figure(figsize=(15, 6))
    for f in features:
        sns.lineplot(x=date_column, y=f'{f}_rolling_mean', data=rolling_data, label=f'{f} (rolling {window}d)')
    plt.title(f'{window}-Day Rolling Mean of Features')
    plt.xlabel('Date')
    plt.ylabel('Feature Value (Rolling Mean)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/rolling_features.png')
    plt.close()
    print("Rolling features plot saved to plots/rolling_features.png")

def main():
    # Define file paths
    data_dir = 'data/processed/merged'
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Specify the ticker files to analyze
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    
    for ticker in tickers:
        filepath = os.path.join(data_dir, f"{ticker}_merged.csv")
        data = load_data(filepath)
        
        if data is None:
            continue
        
        print(f"\nPerforming EDA for {ticker}...")
        
        # Check for missing values
        check_missing_values(data)
        
        # Identify feature columns (excluding Date and Target)
        feature_columns = list(data.columns)
        if 'Date' in feature_columns:
            feature_columns.remove('Date')
        if 'Target' in feature_columns:
            feature_columns.remove('Target')
        
        # Plot feature distributions
        plot_feature_distributions(data, feature_columns)
        
        # Plot correlation matrix
        plot_correlation_matrix(data, feature_columns)
        
        # Plot target distribution
        plot_target_distribution(data)
        
        # Plot features vs target using violin plots for better distribution insights
        plot_features_vs_target(data, feature_columns)
        
        # Identify categorical features
        categorical_features = [col for col in data.columns if 'day' in col.lower() or 'month' in col.lower()]
        if categorical_features:
            plot_categorical_features(data, categorical_features)
        
        # Plot time series for selected features
        time_series_features = ['MACD', 'RSI', 'average_sentiment', 'mention_count']
        available_time_series_features = [f for f in time_series_features if f in data.columns]
        if 'Date' in data.columns and len(available_time_series_features) > 0:
            plot_time_series(data, 'Date', available_time_series_features)
            plot_rolling_features(data, 'Date', available_time_series_features, window=30)
        
        # Plot feature correlation with target (if binary)
        if 'Target' in data.columns and data['Target'].nunique() == 2:
            plot_feature_correlation_with_target(data, feature_columns, target_col='Target')
            plot_feature_pairplots(data, feature_columns, target_col='Target', top_n=5)
        
        print(f"EDA completed for {ticker}.\n{'-'*50}")

if __name__ == "__main__":
    main()
