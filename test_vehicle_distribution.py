"""
Test script to analyze vehicle distributions in the ride-hailing simulation data.
This script analyzes the distribution of vehicle types and powertrains.
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def load_json_file(filepath):
    """Load JSON file into a Python object"""
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze_vehicle_distributions(static_data, dynamic_data):
    """Analyze vehicle type and powertrain distributions"""
    # Convert to pandas DataFrames
    static_df = pd.DataFrame(static_data)
    dynamic_df = pd.DataFrame(dynamic_data)
    
    # Merge static and dynamic data
    df = pd.merge(static_df, dynamic_df, on='driver_id')
    
    # Vehicle type distribution
    vehicle_type_stats = df['vehicle_type'].value_counts().to_dict()
    vehicle_type_percentages = (df['vehicle_type'].value_counts(normalize=True) * 100).round(2).to_dict()
    
    # Powertrain distribution
    powertrain_stats = df['vehicle_powertrain'].value_counts().to_dict()
    powertrain_percentages = (df['vehicle_powertrain'].value_counts(normalize=True) * 100).round(2).to_dict()
    
    # Powertrain distribution by vehicle type
    powertrain_by_type = df.groupby(['vehicle_type', 'vehicle_powertrain']).size().unstack(fill_value=0)
    powertrain_by_type_pct = powertrain_by_type.div(powertrain_by_type.sum(axis=1), axis=0) * 100
    
    # Average rating by powertrain
    rating_by_powertrain = df.groupby('vehicle_powertrain')['rating'].agg(['mean', 'std']).round(2).to_dict('index')
    
    # Money earned by powertrain
    earnings_by_powertrain = df.groupby('vehicle_powertrain')['money_earned'].agg(['mean', 'std']).round(2).to_dict('index')
    
    return {
        'vehicle_type_stats': vehicle_type_stats,
        'vehicle_type_percentages': vehicle_type_percentages,
        'powertrain_stats': powertrain_stats,
        'powertrain_percentages': powertrain_percentages,
        'powertrain_by_type': powertrain_by_type.to_dict(),
        'powertrain_by_type_pct': powertrain_by_type_pct.to_dict(),
        'rating_by_powertrain': rating_by_powertrain,
        'earnings_by_powertrain': earnings_by_powertrain,
        'data': df
    }

def plot_distributions(analysis, output_dir):
    """Generate plots for vehicle distributions"""
    # Vehicle type and powertrain distributions
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Vehicle types
    pd.Series(analysis['vehicle_type_percentages']).plot(kind='bar', ax=ax1)
    ax1.set_title('Vehicle Type Distribution')
    ax1.set_ylabel('Percentage')
    ax1.tick_params(axis='x', rotation=45)
    
    # Powertrains
    pd.Series(analysis['powertrain_percentages']).plot(kind='bar', ax=ax2)
    ax2.set_title('Powertrain Distribution')
    ax2.set_ylabel('Percentage')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vehicle_distributions.png'))
    plt.close()
    
    # Powertrain distribution by vehicle type
    plt.figure(figsize=(12, 6))
    df_pct = pd.DataFrame(analysis['powertrain_by_type_pct'])
    df_pct.plot(kind='bar', stacked=True)
    plt.title('Powertrain Distribution by Vehicle Type')
    plt.xlabel('Vehicle Type')
    plt.ylabel('Percentage')
    plt.legend(title='Powertrain')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'powertrain_by_type.png'))
    plt.close()
    
    # Ratings and earnings by powertrain
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Ratings
    sns.boxplot(data=analysis['data'], x='vehicle_powertrain', y='rating', ax=ax1)
    ax1.set_title('Driver Ratings by Powertrain')
    ax1.tick_params(axis='x', rotation=45)
    
    # Earnings
    sns.boxplot(data=analysis['data'], x='vehicle_powertrain', y='money_earned', ax=ax2)
    ax2.set_title('Driver Earnings by Powertrain')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_by_powertrain.png'))
    plt.close()

def main():
    # Define directories
    data_dir = './data_with_age'
    analysis_dir = './vehicle_analysis'
    
    # Create analysis directory
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Generate new data
    print("Generating new data with vehicle distributions...")
    os.system(f'python generate_static_data.py --output_dir {data_dir}')
    
    # Load and analyze data
    print("\nAnalyzing vehicle data...")
    drivers_static = load_json_file(os.path.join(data_dir, 'drivers_static.json'))
    drivers_dynamic = load_json_file(os.path.join(data_dir, 'drivers_dynamic.json'))
    
    # Analyze distributions
    analysis = analyze_vehicle_distributions(drivers_static, drivers_dynamic)
    
    # Generate plots
    print("\nGenerating visualization plots...")
    plot_distributions(analysis, analysis_dir)
    
    # Print statistics
    print("\nVehicle Type Distribution (%):")
    print(json.dumps(analysis['vehicle_type_percentages'], indent=2))
    
    print("\nPowertrain Distribution (%):")
    print(json.dumps(analysis['powertrain_percentages'], indent=2))
    
    print("\nAverage Rating by Powertrain:")
    print(json.dumps(analysis['rating_by_powertrain'], indent=2))
    
    print("\nAverage Earnings by Powertrain:")
    print(json.dumps(analysis['earnings_by_powertrain'], indent=2))
    
    print("\nAnalysis complete! Check the 'vehicle_analysis' directory for visualizations.")

if __name__ == "__main__":
    main() 