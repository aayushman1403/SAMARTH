"""
Crime Case Prioritization System
--------------------------------
This script generates a synthetic dataset of crime cases, analyzes them using sentiment analysis
and machine learning, and outputs the top 10 highest priority cases that need urgent attention.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import joblib
import nltk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Download all necessary NLTK resources
print("Downloading necessary NLTK resources...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('wordnet', quiet=True)

# Import project modules
from data_generator import DataGenerator
from data_preprocessor import DataPreprocessor
from sentiment_analyzer import SentimentAnalyzer
from feature_engineering import FeatureEngineer
from model_builder import ModelBuilder

def main():
    print("\n" + "="*80)
    print("                      CRIME CASE PRIORITIZATION SYSTEM")
    print("="*80 + "\n")
    
    # Step 1: Generate synthetic crime data
    print("STEP 1: Generating synthetic crime data...\n")
    try:
        generator = DataGenerator(num_samples=1000, random_state=42)
        df = generator.generate_dataset()
        
        # Save to CSV instead of Excel to avoid potential issues
        csv_filename = 'crime_data_synthetic.csv'
        df.to_csv(csv_filename, index=False)
        print(f"Dataset generated successfully with 1000 samples and saved to {csv_filename}\n")
    except Exception as e:
        print(f"Error generating data: {e}")
        return
    
    # Display some basic statistics about the data
    print(f"Generated {len(df)} crime cases with the following distribution:")
    crime_counts = df['CrimeType'].value_counts()
    for crime_type, count in crime_counts.items():
        print(f"  - {crime_type}: {count} cases ({count/len(df)*100:.1f}%)")
    
    priority_counts = df['PriorityLabel'].value_counts()
    print(f"\nPriority distribution:")
    print(f"  - High Priority: {priority_counts.get(1, 0)} cases ({priority_counts.get(1, 0)/len(df)*100:.1f}%)")
    print(f"  - Low Priority: {priority_counts.get(0, 0)} cases ({priority_counts.get(0, 0)/len(df)*100:.1f}%)")
    
    # Step 2: Preprocess data
    print("\nSTEP 2: Preprocessing data...")
    try:
        preprocessor = DataPreprocessor()
        processed_df = preprocessor.preprocess_data(df)
        print(f"Data preprocessing completed successfully.\n")
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return
    
    # Step 3: Perform sentiment analysis
    print("STEP 3: Performing sentiment analysis...")
    try:
        sentiment_analyzer = SentimentAnalyzer()
        sentiment_df = sentiment_analyzer.add_sentiment_features(processed_df)
        print(f"Sentiment analysis completed successfully.\n")
        
        # Print sentiment summary
        print("Sentiment analysis summary:")
        sentiment_categories = sentiment_df['sentiment_category'].value_counts()
        for category, count in sentiment_categories.items():
            print(f"  - {category.title()} sentiment: {count} cases ({count/len(sentiment_df)*100:.1f}%)")
        
        # Sentiment correlation with priority
        sentiment_corr = sentiment_df[['sentiment_compound', 'PriorityLabel']].corr()
        print(f"\nCorrelation between sentiment and priority: {sentiment_corr.iloc[0,1]:.3f}")
    except Exception as e:
        print(f"Error performing sentiment analysis: {e}")
        return
    
    # Step 4: Engineer features
    print("\nSTEP 4: Engineering features...")
    try:
        feature_engineer = FeatureEngineer()
        engineered_df = feature_engineer.engineer_features(sentiment_df)
        print(f"Feature engineering completed successfully.\n")
    except Exception as e:
        print(f"Error engineering features: {e}")
        return
    
    # Step 5: Build and train model
    print("STEP 5: Building and training the model...")
    try:
        X, y = feature_engineer.prepare_model_input(engineered_df)
        model_builder = ModelBuilder()
        X_train, X_test, y_train, y_test = model_builder.split_data(X, y)
        model = model_builder.build_model(X_train, y_train, tune_hyperparameters=False)
        
        # Evaluate model
        metrics = model_builder.evaluate_model(X_test, y_test)
        feature_importances = model_builder.get_feature_importances()
        
        print("\nTop 10 features for determining case priority:")
        for i, (feature, importance) in enumerate(zip(feature_importances['feature'].head(10), 
                                                    feature_importances['importance'].head(10)), 1):
            print(f"  {i}. {feature}: {importance:.4f}")
    except Exception as e:
        print(f"Error building model: {e}")
        return
    
    # Step 6: Generate priority scores and identify top cases
    print("\nSTEP 6: Identifying high priority cases...")
    try:
        # Predict priorities for all cases
        predictions, probabilities = model_builder.predict_priority(X)
        
        # Create a results dataframe
        results_df = processed_df[['CaseID', 'FilingDate', 'CrimeType', 'Location', 'Description']].copy()
        results_df['PriorityScore'] = probabilities
        results_df['PriorityLabel'] = predictions
        
        # Add filing recency as days since filed
        results_df['DaysSinceFiled'] = (datetime.now().date() - results_df['FilingDate'].dt.date).dt.days
        
        # Sort by priority score (descending)
        results_df = results_df.sort_values(by='PriorityScore', ascending=False)
        
        # Save full results to CSV
        results_csv = 'crime_cases_prioritized.csv'
        results_df.to_csv(results_csv, index=False)
        
        # Try to save to Excel if openpyxl is available
        try:
            results_excel = 'crime_cases_prioritized.xlsx'
            results_df.to_excel(results_excel, index=False)
            print(f"All cases with priority scores saved to {results_excel}")
        except:
            print(f"All cases with priority scores saved to {results_csv}")
    except Exception as e:
        print(f"Error prioritizing cases: {e}")
        return
    
    # Display top 10 high priority cases
    print("\n" + "="*80)
    print("                   TOP 10 CASES REQUIRING IMMEDIATE ATTENTION")
    print("="*80)
    
    top_cases = results_df.head(10)
    
    for i, case in enumerate(top_cases.itertuples(), 1):
        print(f"\nCASE {i} - PRIORITY SCORE: {case.PriorityScore*100:.1f}%")
        print(f"Case ID: {case.CaseID}")
        print(f"Filing Date: {case.FilingDate.strftime('%Y-%m-%d')} ({case.DaysSinceFiled} days ago)")
        print(f"Crime Type: {case.CrimeType}")
        print(f"Location: {case.Location}")
        
        # Truncate description if it's too long
        desc = case.Description
        if len(desc) > 150:
            desc = desc[:150] + "..."
        print(f"Description: {desc}")
        print("-" * 80)
    
    # Create a summary of high priority crime types
    high_priority = results_df[results_df['PriorityLabel'] == 1]
    high_priority_crimes = high_priority['CrimeType'].value_counts()
    
    print("\nSUMMARY OF HIGH PRIORITY CASES:")
    print(f"Total high priority cases identified: {len(high_priority)} out of {len(results_df)} ({len(high_priority)/len(results_df)*100:.1f}%)")
    print("\nDistribution by crime type:")
    for crime_type, count in high_priority_crimes.items():
        print(f"  - {crime_type}: {count} cases ({count/len(high_priority)*100:.1f}% of high priority)")
    
    # Generate a simple visualization of high priority cases by crime type
    try:
        # Plot crime type distribution in high priority cases
        plt.figure(figsize=(12, 6))
        ax = high_priority_crimes.plot(kind='bar')
        plt.title('Distribution of Crime Types in High Priority Cases')
        plt.xlabel('Crime Type')
        plt.ylabel('Number of High Priority Cases')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        chart_file = 'high_priority_crime_distribution.png'
        plt.savefig(chart_file)
        print(f"\nA chart of high priority cases by crime type has been saved to {chart_file}")
    except Exception as e:
        print(f"Could not generate visualization: {e}")
    
    print("\n" + "="*80)
    print("                       PRIORITIZATION ANALYSIS COMPLETE")
    print("="*80)
    print("\nThe full prioritized case list can be found in 'crime_cases_prioritized.csv'")

if __name__ == "__main__":
    main()