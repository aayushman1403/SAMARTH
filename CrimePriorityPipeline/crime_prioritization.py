import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Import project modules
from data_generator import DataGenerator
from data_preprocessor import DataPreprocessor
from sentiment_analyzer import SentimentAnalyzer
from feature_engineering import FeatureEngineer
from model_builder import ModelBuilder
from utils import plot_feature_importances

def main():
    print("Crime Case Prioritization System")
    print("===============================\n")
    
    # Step 1: Generate synthetic crime data
    print("Step 1: Generating synthetic crime data...")
    generator = DataGenerator(num_samples=1000, random_state=42)
    df = generator.generate_dataset()
    filename = generator.save_dataset(df)
    print(f"Dataset generated successfully with 1000 samples and saved to {filename}\n")
    
    # Step 2: Preprocess data
    print("Step 2: Preprocessing data...")
    preprocessor = DataPreprocessor()
    processed_df = preprocessor.preprocess_data(df)
    print(f"Data preprocessing completed. Processed dataframe shape: {processed_df.shape}\n")
    
    # Step 3: Perform sentiment analysis
    print("Step 3: Performing sentiment analysis...")
    sentiment_analyzer = SentimentAnalyzer()
    sentiment_df = sentiment_analyzer.add_sentiment_features(processed_df)
    print(f"Sentiment analysis completed. Dataframe shape: {sentiment_df.shape}\n")
    
    # Step 4: Engineer features
    print("Step 4: Engineering features...")
    feature_engineer = FeatureEngineer()
    engineered_df = feature_engineer.engineer_features(sentiment_df)
    print(f"Feature engineering completed. Dataframe shape: {engineered_df.shape}\n")
    
    # Step 5: Build and train model
    print("Step 5: Building and training the model...")
    X, y = feature_engineer.prepare_model_input(engineered_df)
    model_builder = ModelBuilder()
    X_train, X_test, y_train, y_test = model_builder.split_data(X, y)
    model = model_builder.build_model(X_train, y_train, tune_hyperparameters=False)
    
    # Step 6: Evaluate model
    print("Step 6: Evaluating the model...")
    metrics = model_builder.evaluate_model(X_test, y_test)
    feature_importances = model_builder.get_feature_importances()
    print("\nTop 10 Feature Importances:")
    print(feature_importances.head(10))
    
    # Save model
    model_path = model_builder.save_model()
    print(f"\nModel saved to {model_path}\n")
    
    # Step 7: Generate priority scores for all cases
    print("Step 7: Generating priority scores for all cases...")
    # Add model predictions back to the original dataframe
    predictions, probabilities = model_builder.predict_priority(X)
    
    # Create a copy of the original dataframe with key fields
    results_df = processed_df[['CaseID', 'FilingDate', 'CrimeType', 'Location', 'Description']].copy()
    results_df['ActualPriority'] = processed_df['PriorityLabel']
    results_df['PredictedPriority'] = predictions
    results_df['PriorityScore'] = probabilities  # Probability of being high priority
    
    # Sort by priority score (descending)
    results_df = results_df.sort_values(by='PriorityScore', ascending=False)
    
    # Save full results to Excel
    results_df.to_excel('crime_cases_prioritized.xlsx', index=False)
    print(f"All cases with priority scores saved to crime_cases_prioritized.xlsx\n")
    
    # Step 8: Display top 10 high priority cases
    print("Step 8: Top 10 cases requiring immediate attention:")
    print("==================================================")
    top_cases = results_df.head(10)
    
    for i, case in enumerate(top_cases.itertuples(), 1):
        print(f"\nCase {i}:")
        print(f"Case ID: {case.CaseID}")
        print(f"Filing Date: {case.FilingDate.strftime('%Y-%m-%d')}")
        print(f"Crime Type: {case.CrimeType}")
        print(f"Location: {case.Location}")
        print(f"Priority Score: {case.PriorityScore:.4f} ({case.PriorityScore*100:.1f}%)")
        print(f"Description: {case.Description[:150]}..." if len(case.Description) > 150 else f"Description: {case.Description}")
        print("-" * 80)
    
    # Step 9: Generate and save additional analysis
    print("\nStep 9: Generating additional analysis...")
    
    # Analyze crime types among high priority cases
    high_priority_crimes = results_df[results_df['PredictedPriority'] == 1]['CrimeType'].value_counts()
    
    # Plot crime type distribution in high priority cases
    plt.figure(figsize=(12, 6))
    high_priority_crimes.plot(kind='bar')
    plt.title('Distribution of Crime Types in High Priority Cases')
    plt.xlabel('Crime Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('high_priority_crime_types.png')
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    fig = plot_feature_importances(feature_importances)
    plt.tight_layout()
    plt.savefig('feature_importances.png')
    
    # Calculate average time since filing for high priority cases
    if 'days_since_filing' in engineered_df.columns:
        high_priority_days = engineered_df[engineered_df['PriorityLabel'] == 1]['days_since_filing'].mean()
        low_priority_days = engineered_df[engineered_df['PriorityLabel'] == 0]['days_since_filing'].mean()
        print(f"\nAverage days since filing for high priority cases: {high_priority_days:.1f} days")
        print(f"Average days since filing for low priority cases: {low_priority_days:.1f} days")
    
    # Sentiment analysis across priority levels
    if all(col in sentiment_df.columns for col in ['sentiment_compound', 'PriorityLabel']):
        high_sentiment = sentiment_df[sentiment_df['PriorityLabel'] == 1]['sentiment_compound'].mean()
        low_sentiment = sentiment_df[sentiment_df['PriorityLabel'] == 0]['sentiment_compound'].mean()
        print(f"Average sentiment score for high priority cases: {high_sentiment:.4f}")
        print(f"Average sentiment score for low priority cases: {low_sentiment:.4f}")
    
    print("\nCrime Case Prioritization completed successfully!")
    print("Check the Excel file 'crime_cases_prioritized.xlsx' for complete results")
    print("Analysis charts saved to 'high_priority_crime_types.png' and 'feature_importances.png'")

if __name__ == "__main__":
    main()