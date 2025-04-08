"""
Test script to analyze age and gender distributions in the ride-hailing simulation data.
This script compares data between two directories and generates statistics about demographic distributions.
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import numpy as np

def load_json_file(filepath):
    """Load JSON file into a Python object"""
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze_distributions(static_data, dynamic_data):
    """Analyze age and gender distributions and correlations"""
    # Convert to pandas DataFrames
    static_df = pd.DataFrame(static_data)
    dynamic_df = pd.DataFrame(dynamic_data)
    
    # Merge static and dynamic data
    df = pd.merge(static_df, dynamic_df, on='user_id' if 'user_id' in static_df else 'driver_id')
    
    # Basic age statistics
    age_stats = {
        'mean': df['age'].mean(),
        'median': df['age'].median(),
        'std': df['age'].std(),
        'min': df['age'].min(),
        'max': df['age'].max()
    }
    
    # Gender distribution
    gender_stats = df['gender'].value_counts().to_dict()
    gender_percentages = (df['gender'].value_counts(normalize=True) * 100).round(2).to_dict()
    
    # Age correlations with other numeric columns
    correlations = {}
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        if col != 'age':
            correlations[col] = df['age'].corr(df[col])
    
    # Age distribution by gender
    age_by_gender = df.groupby('gender')['age'].agg(['mean', 'median', 'std']).to_dict('index')
    
    return {
        'age_stats': age_stats,
        'gender_stats': gender_stats,
        'gender_percentages': gender_percentages,
        'age_by_gender': age_by_gender,
        'correlations': correlations,
        'data': df
    }

def plot_distributions(users_analysis, drivers_analysis, output_dir):
    """Generate plots for age and gender distributions"""
    # Age distributions
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    sns.kdeplot(data=users_analysis['data'], x='age', label='Users')
    sns.kdeplot(data=drivers_analysis['data'], x='age', label='Drivers')
    plt.title('Age Distribution: Users vs Drivers')
    plt.xlabel('Age')
    plt.ylabel('Density')
    
    # Gender distributions
    plt.subplot(2, 1, 2)
    x = np.arange(len(users_analysis['gender_stats']))
    width = 0.35
    
    plt.bar(x - width/2, users_analysis['gender_percentages'].values(), width, label='Users')
    plt.bar(x + width/2, drivers_analysis['gender_percentages'].values(), width, label='Drivers')
    plt.xticks(x, users_analysis['gender_stats'].keys())
    plt.title('Gender Distribution: Users vs Drivers')
    plt.ylabel('Percentage')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'demographic_distributions.png'))
    plt.close()
    
    # Age distribution by gender
    plt.figure(figsize=(15, 5))
    
    # Users
    plt.subplot(1, 2, 1)
    sns.boxplot(data=users_analysis['data'], x='gender', y='age')
    plt.title('User Age Distribution by Gender')
    
    # Drivers
    plt.subplot(1, 2, 2)
    sns.boxplot(data=drivers_analysis['data'], x='gender', y='age')
    plt.title('Driver Age Distribution by Gender')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'age_by_gender.png'))
    plt.close()
    
    # Correlation heatmaps
    plt.figure(figsize=(15, 6))
    
    # Users correlations
    plt.subplot(1, 2, 1)
    user_corr = pd.DataFrame(users_analysis['correlations'].items(), columns=['Metric', 'Correlation'])
    sns.barplot(data=user_corr, x='Metric', y='Correlation')
    plt.title('User Age Correlations')
    plt.xticks(rotation=45)
    
    # Drivers correlations
    plt.subplot(1, 2, 2)
    driver_corr = pd.DataFrame(drivers_analysis['correlations'].items(), columns=['Metric', 'Correlation'])
    sns.barplot(data=driver_corr, x='Metric', y='Correlation')
    plt.title('Driver Age Correlations')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'age_correlations.png'))
    plt.close()

def main():
    # Define directories
    original_dir = './data'
    new_dir = './data_with_age'
    analysis_dir = './age_analysis'
    
    # Create analysis directory
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Generate new data
    print("Generating new data with demographic distributions...")
    os.system(f'python generate_static_data.py --output_dir {new_dir}')
    
    # Load and analyze new data
    print("\nAnalyzing new data...")
    users_static = load_json_file(os.path.join(new_dir, 'users_static.json'))
    users_dynamic = load_json_file(os.path.join(new_dir, 'users_dynamic.json'))
    drivers_static = load_json_file(os.path.join(new_dir, 'drivers_static.json'))
    drivers_dynamic = load_json_file(os.path.join(new_dir, 'drivers_dynamic.json'))
    
    # Analyze distributions
    users_analysis = analyze_distributions(users_static, users_dynamic)
    drivers_analysis = analyze_distributions(drivers_static, drivers_dynamic)
    
    # Generate plots
    print("\nGenerating visualization plots...")
    plot_distributions(users_analysis, drivers_analysis, analysis_dir)
    
    # Print statistics
    print("\nUser Demographics:")
    print("Age Statistics:")
    print(json.dumps(users_analysis['age_stats'], indent=2))
    print("\nGender Distribution (%):")
    print(json.dumps(users_analysis['gender_percentages'], indent=2))
    print("\nAge by Gender:")
    print(json.dumps(users_analysis['age_by_gender'], indent=2))
    
    print("\nDriver Demographics:")
    print("Age Statistics:")
    print(json.dumps(drivers_analysis['age_stats'], indent=2))
    print("\nGender Distribution (%):")
    print(json.dumps(drivers_analysis['gender_percentages'], indent=2))
    print("\nAge by Gender:")
    print(json.dumps(drivers_analysis['age_by_gender'], indent=2))
    
    print("\nAnalysis complete! Check the 'age_analysis' directory for visualizations.")

if __name__ == "__main__":
    main() 