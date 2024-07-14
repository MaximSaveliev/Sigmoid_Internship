import pandas as pd
import numpy as np
import dask
import dask.dataframe as dd
import matplotlib.pyplot as plt
import hvplot.dask
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest

def analyze_nan(df : pd.DataFrame):
    nan_count = df.isna().sum()
    nan_percent = 100 * df.isna().sum() / len(df)
    nan_summary = pd.concat([nan_count, nan_percent], axis=1, keys=['NaN Count', 'NaN Percent'])
    return nan_summary.sort_values('NaN Percent', ascending=False)

def delete_high_nan_columns(df : pd.DataFrame, nan_summary, threshold=90):
    # Get the names of columns with NaN percentage >= threshold
    columns_to_drop = nan_summary[nan_summary['NaN Percent'] >= threshold].index.tolist()
    
    # Print the columns that will be dropped
    print(f"Dropping {len(columns_to_drop)} columns:")
    for col in columns_to_drop:
        print(f"- {col}: {nan_summary.loc[col, 'NaN Percent']:.2f}% NaN")
    
    # Drop the columns
    df_cleaned = df.drop(columns=columns_to_drop)
    
    print(f"\nOriginal DataFrame shape: {df.shape}")
    print(f"Cleaned DataFrame shape: {df_cleaned.shape}")
    
    return df_cleaned

def clean_numeric_columns_with_spaces(df : pd.DataFrame, columns):
    for column in columns:
        if column in df.columns:
            # Remove spaces from the values
            df[column] = df[column].astype(str).str.replace(' ', '')
        else:
            print(f"Column {column} not found in the DataFrame.")
    
    return df

def convert_numeric_columns(df : pd.DataFrame, columns):
    for column in columns:
        if column in df.columns:
            # Try to convert to float first to handle any string representations of floats
            df[column] = pd.to_numeric(df[column], errors='coerce')
            
            # Check if all non-null values can be converted to int
            mask = df[column].notnull() & (df[column] % 1 != 0)
            problematic_values = df.loc[mask, column]
            
            if problematic_values.empty:
                # All non-null values are integers
                df[column] = df[column].astype('Int64')
                # print(f"Converted {column} to Int64")
            else:
                df[column] = df[column].astype('float64')
                # print(f"Converted {column} to float64")
        else:
            print(f"Column {column} not found in the DataFrame.")
    
    return df

def label_encode_series(df: pd.DataFrame, columns):
    for col in columns:
        if col in df.columns:
            if df[col].dtype == 'object':  # Encode only object dtype columns
                label_encoder = LabelEncoder()
                non_null_mask = df[col].notnull()
                df.loc[non_null_mask, col] = label_encoder.fit_transform(df.loc[non_null_mask, col])
        else:
            print(f"Column {col} not found in the DataFrame.")
    return df

def plot_imputed_scatter(initial_df: pd.DataFrame, imputed_df: pd.DataFrame, columns_to_plot, figsize=(30, 20)):
    # Ensure all specified columns exist in both dataframes
    for col in columns_to_plot:
        if col not in initial_df.columns or col not in imputed_df.columns:
            raise ValueError(f"Column '{col}' not found in one or both dataframes")
    
    # Create subplots
    fig, axes = plt.subplots(len(columns_to_plot), 1, figsize=figsize)
    if len(columns_to_plot) == 1:
        axes = [axes]
    
    for i, col in enumerate(columns_to_plot):
        ax = axes[i]
        
        # Plot original non-NaN values
        mask = ~initial_df[col].isna()
        ax.scatter(initial_df.index[mask], initial_df[col][mask], 
                   color='blue', alpha=0.5, label='Original Data')
        
        # Plot imputed values (only those that were originally NaN)
        mask_imputed = initial_df[col].isna()
        ax.scatter(imputed_df.index[mask_imputed], imputed_df[col][mask_imputed], 
                   color='green', alpha=0.5, label='Imputed Data')
        
        # Calculate IQR (Interquartile range) for ylim setting
        data_combined = pd.concat([initial_df[col][mask], imputed_df[col][mask_imputed]])
        Q1 = data_combined.quantile(0.25)
        Q3 = data_combined.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        ax.set_ylim(lower_bound, upper_bound)
        
        ax.set_title(f'Scatter Plot of {col}')
        ax.set_xlabel('Index')
        ax.set_ylabel(col)
        ax.legend()
    
    plt.tight_layout()
    plt.show()

def handle_negative_values(ddf, columns_to_check):
    for col in columns_to_check:
        ddf[col] = ddf[col].map_partitions(lambda partition: partition.abs(), meta=(col, 'float64'))
    return ddf

def plot_outliers(df, feature_column, contaminations_dict):
    # Initialize Isolation Forest model for the specific feature column
    model = IsolationForest(contamination=contaminations_dict[feature_column])

    # Fit the model to detect outliers for the specific feature column
    X = df[[feature_column]]
    model.fit(X)

    # Predict outliers (-1 for outliers, 1 for inliers)
    df['is_outlier'] = model.predict(X)
    df['is_outlier'] = df['is_outlier'].astype('category')

    # Plotting
    colors = sns.color_palette("pastel")

    # Plot 1: Without Outliers
    plt.subplot(1, 3, 2)
    sns.scatterplot(x=df[df['is_outlier'] == 1].index, y=df[df['is_outlier'] == 1][feature_column], color=colors[1])
    plt.title('Without Outliers')
    plt.xlabel('Index')
    plt.ylabel(feature_column)

    # Plot 2: Only Outliers
    plt.subplot(1, 3, 3)
    sns.scatterplot(x=df[df['is_outlier'] == -1].index, y=df[df['is_outlier'] == -1][feature_column], color=colors[2])
    plt.title('Only Outliers')
    plt.xlabel('Index')
    plt.ylabel(feature_column)

    # Adjust plot size and layout
    plt.gcf().set_size_inches(15, 5)
    plt.suptitle(f'{feature_column} with Outliers Analysis', fontsize=16, y=1)

    # Save plots to files instead of showing them
    plt.tight_layout()
    plt.show()

